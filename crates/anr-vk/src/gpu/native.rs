use anr_core::Result;
use anr_dfn3::{Model, PARAMETER_BYTES, STATE_SIZE, backend};
use ash::{Entry, vk};
use std::{ffi::CStr, ptr};

pub(super) fn checked<T>(operation: &str, result: std::result::Result<T, vk::Result>) -> Result<T> {
    result.map_err(|error| backend(format!("Vulkan {operation}: {error:?}")))
}

struct Instance {
    instance: ash::Instance,
    // Keep the dynamic loader alive until after vkDestroyInstance.
    _entry: Entry,
}
impl Drop for Instance {
    fn drop(&mut self) {
        unsafe { self.instance.destroy_instance(None) };
    }
}

pub(super) struct Context {
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub family: u32,
    pub limits: vk::PhysicalDeviceLimits,
    pub memory: vk::PhysicalDeviceMemoryProperties,
    _instance: Instance,
}
impl Context {
    pub fn new(model: &Model) -> Result<Self> {
        let entry = unsafe { Entry::load() }
            .map_err(|error| backend(format!("loading native Vulkan loader: {error}")))?;
        let version = checked("enumerate instance version", unsafe {
            entry.try_enumerate_instance_version()
        })?
        .unwrap_or(vk::API_VERSION_1_0);
        if version < vk::API_VERSION_1_1 {
            return Err(backend("Vulkan loader does not support Vulkan 1.1"));
        }
        let app = vk::ApplicationInfo::default()
            .application_name(c"anr-vk")
            .api_version(vk::API_VERSION_1_1);
        let instance = checked("create instance", unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
        })?;
        let instance = Instance {
            instance,
            _entry: entry,
        };
        let mut devices = checked("enumerate physical devices", unsafe {
            instance.instance.enumerate_physical_devices()
        })?;
        // Prefer discrete hardware; neither CPU nor unclassified software devices
        // are eligible. No layers, surfaces, extensions, or optional features.
        devices.sort_by_key(|&device| {
            let properties = unsafe { instance.instance.get_physical_device_properties(device) };
            match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                _ => 3,
            }
        });
        let mut reasons = Vec::new();
        for physical in devices {
            let properties = unsafe { instance.instance.get_physical_device_properties(physical) };
            let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }.to_string_lossy();
            if !matches!(
                properties.device_type,
                vk::PhysicalDeviceType::DISCRETE_GPU
                    | vk::PhysicalDeviceType::INTEGRATED_GPU
                    | vk::PhysicalDeviceType::VIRTUAL_GPU
            ) {
                reasons.push(format!(
                    "{name}: not a hardware GPU ({:?})",
                    properties.device_type
                ));
                continue;
            }
            if properties.api_version < vk::API_VERSION_1_1 {
                reasons.push(format!("{name}: Vulkan 1.1 is required"));
                continue;
            }
            if let Err(error) = validate_limits(&properties.limits, model) {
                reasons.push(format!("{name}: {error}"));
                continue;
            }
            let families = unsafe {
                instance
                    .instance
                    .get_physical_device_queue_family_properties(physical)
            };
            let family = families
                .iter()
                .enumerate()
                .filter(|(_, p)| {
                    p.queue_count != 0 && p.queue_flags.contains(vk::QueueFlags::COMPUTE)
                })
                .min_by_key(|(_, p)| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .map(|(i, _)| i as u32);
            let Some(family) = family else {
                reasons.push(format!("{name}: no compute queue"));
                continue;
            };
            let priorities = [1.0];
            let queue = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priorities)];
            let device = match unsafe {
                instance.instance.create_device(
                    physical,
                    &vk::DeviceCreateInfo::default().queue_create_infos(&queue),
                    None,
                )
            } {
                Ok(device) => device,
                Err(error) => {
                    reasons.push(format!("{name}: vkCreateDevice: {error:?}"));
                    continue;
                }
            };
            let queue = unsafe { device.get_device_queue(family, 0) };
            let memory = unsafe {
                instance
                    .instance
                    .get_physical_device_memory_properties(physical)
            };
            return Ok(Self {
                device,
                queue,
                family,
                limits: properties.limits,
                memory,
                _instance: instance,
            });
        }
        Err(backend(format!(
            "no usable hardware Vulkan 1.1 compute device: {}",
            reasons.join("; ")
        )))
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        // No work is submitted before Gpu exists, and Gpu drains its queue
        // before dropping any resources or this device owner.
        unsafe { self.device.destroy_device(None) };
    }
}

fn validate_limits(limits: &vk::PhysicalDeviceLimits, model: &Model) -> Result<()> {
    if limits.max_per_stage_descriptor_storage_buffers < 4
        || limits.max_descriptor_set_storage_buffers < 4
        || limits.max_per_stage_descriptor_uniform_buffers < 1
        || limits.max_descriptor_set_uniform_buffers_dynamic < 1
        || limits.max_uniform_buffer_range < PARAMETER_BYTES
        || limits.max_compute_work_group_invocations < 128
        || limits.max_compute_work_group_size[0] < 128
    {
        return Err(backend(
            "insufficient compute/storage/dynamic-uniform limits for DFN3",
        ));
    }
    for size in [
        model.weights.len() as u64 * 4,
        STATE_SIZE as u64 * 4,
        u64::from(model.scratch_len) * 4,
    ] {
        if size > u64::from(limits.max_storage_buffer_range) {
            return Err(backend(format!(
                "storage buffer requires {size} bytes; limit is {}",
                limits.max_storage_buffer_range
            )));
        }
    }
    if model
        .dispatches
        .iter()
        .any(|step| step.groups > limits.max_compute_work_group_count[0])
    {
        return Err(backend("dispatch exceeds maxComputeWorkGroupCount[0]"));
    }
    Ok(())
}

// Every resource owns a dispatch-table clone, not the logical device. Its
// Context must outlive it (Gpu declares Context last; constructors create it
// first). This guard also covers every intermediate construction failure.
pub(super) struct Resource<T: Copy> {
    pub value: T,
    device: ash::Device,
    destroy: fn(&ash::Device, T),
}
impl<T: Copy> Resource<T> {
    pub fn new(device: &ash::Device, value: T, destroy: fn(&ash::Device, T)) -> Self {
        Self {
            value,
            device: device.clone(),
            destroy,
        }
    }
}
impl<T: Copy> Drop for Resource<T> {
    fn drop(&mut self) {
        (self.destroy)(&self.device, self.value);
    }
}

pub(super) struct Buffer {
    pub handle: vk::Buffer,
    pub size: u64,
    pub mapped: *mut u8,
    memory: vk::DeviceMemory,
    device: ash::Device,
}
impl Buffer {
    pub fn new(
        context: &Context,
        size: u64,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<Self> {
        let device = &context.device;
        let handle = checked("create buffer", unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        })?;
        let mut buffer = Self {
            handle,
            size,
            mapped: ptr::null_mut(),
            memory: vk::DeviceMemory::null(),
            device: device.clone(),
        };
        let requirements = unsafe { device.get_buffer_memory_requirements(handle) };
        let memory_type = (0..context.memory.memory_type_count)
            .find(|&index| {
                requirements.memory_type_bits & (1 << index) != 0
                    && context.memory.memory_types[index as usize]
                        .property_flags
                        .contains(properties)
            })
            .ok_or_else(|| {
                backend(format!(
                    "no Vulkan memory type for {properties:?} ({size} bytes)"
                ))
            })?;
        buffer.memory = checked("allocate buffer memory", unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )
        })?;
        checked("bind buffer memory", unsafe {
            device.bind_buffer_memory(handle, buffer.memory, 0)
        })?;
        if properties.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            buffer.mapped = checked("map coherent buffer", unsafe {
                device.map_memory(
                    buffer.memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
            })?
            .cast();
        }
        Ok(buffer)
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            if !self.mapped.is_null() {
                self.device.unmap_memory(self.memory);
            }
            self.device.destroy_buffer(self.handle, None);
            if self.memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.memory, None);
            }
        }
    }
}
