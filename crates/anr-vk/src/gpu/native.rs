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
    pub timestamp_valid_bits: u32,
    pub limits: vk::PhysicalDeviceLimits,
    pub memory: vk::PhysicalDeviceMemoryProperties,
    pub cooperative_subgroup_size: Option<u32>,
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
        let api_version = version.min(vk::API_VERSION_1_3);
        let app = vk::ApplicationInfo::default()
            .application_name(c"anr-vk")
            .api_version(api_version);
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
        // are eligible. Optional matrix features never raise the Vulkan 1.1 floor.
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
            let device_api = api_version.min(properties.api_version);
            let cooperative_subgroup_size =
                cooperative_subgroup_size(&instance, physical, device_api)?;
            let extension_names = [
                ash::khr::cooperative_matrix::NAME.as_ptr(),
                ash::ext::subgroup_size_control::NAME.as_ptr(),
                ash::khr::vulkan_memory_model::NAME.as_ptr(),
                ash::khr::shader_float16_int8::NAME.as_ptr(),
            ];
            let mut cooperative =
                vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
            let mut memory_model =
                vk::PhysicalDeviceVulkanMemoryModelFeatures::default().vulkan_memory_model(true);
            let mut float16 =
                vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_float16(true);
            // Naga also declares both 16-bit storage capabilities for workgroup
            // FP16 types, even though our external buffers remain FP32.
            let mut storage16 = vk::PhysicalDevice16BitStorageFeatures::default()
                .storage_buffer16_bit_access(true)
                .uniform_and_storage_buffer16_bit_access(true);
            let mut subgroups = vk::PhysicalDeviceSubgroupSizeControlFeatures::default()
                .compute_full_subgroups(true);
            let mut create_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue);
            if cooperative_subgroup_size.is_some() {
                let extension_count = if device_api >= vk::API_VERSION_1_3 {
                    1
                } else if device_api >= vk::API_VERSION_1_2 {
                    2
                } else {
                    4
                };
                create_info = create_info
                    .enabled_extension_names(&extension_names[..extension_count])
                    .push_next(&mut cooperative)
                    .push_next(&mut memory_model)
                    .push_next(&mut float16)
                    .push_next(&mut storage16)
                    .push_next(&mut subgroups);
            }
            let device = match unsafe {
                instance
                    .instance
                    .create_device(physical, &create_info, None)
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
                timestamp_valid_bits: families[family as usize].timestamp_valid_bits,
                limits: properties.limits,
                memory,
                cooperative_subgroup_size,
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

fn cooperative_subgroup_size(
    instance: &Instance,
    physical: vk::PhysicalDevice,
    api_version: u32,
) -> Result<Option<u32>> {
    let extensions = checked("enumerate device extensions", unsafe {
        instance
            .instance
            .enumerate_device_extension_properties(physical)
    })?;
    let has_extension = |name: &CStr| {
        extensions
            .iter()
            .any(|extension| unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) == name })
    };
    if !has_extension(ash::khr::cooperative_matrix::NAME)
        || (api_version < vk::API_VERSION_1_3
            && !has_extension(ash::ext::subgroup_size_control::NAME))
        || (api_version < vk::API_VERSION_1_2
            && (!has_extension(ash::khr::vulkan_memory_model::NAME)
                || !has_extension(ash::khr::shader_float16_int8::NAME)))
    {
        return Ok(None);
    }
    let mut cooperative = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
    let mut memory_model = vk::PhysicalDeviceVulkanMemoryModelFeatures::default();
    let mut float16 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let mut storage16 = vk::PhysicalDevice16BitStorageFeatures::default();
    let mut subgroups = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut cooperative)
        .push_next(&mut memory_model)
        .push_next(&mut float16)
        .push_next(&mut storage16)
        .push_next(&mut subgroups);
    unsafe {
        instance
            .instance
            .get_physical_device_features2(physical, &mut features)
    };
    if cooperative.cooperative_matrix == vk::FALSE
        || memory_model.vulkan_memory_model == vk::FALSE
        || float16.shader_float16 == vk::FALSE
        || storage16.storage_buffer16_bit_access == vk::FALSE
        || storage16.uniform_and_storage_buffer16_bit_access == vk::FALSE
        || subgroups.compute_full_subgroups == vk::FALSE
    {
        return Ok(None);
    }
    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
    let mut matrix = vk::PhysicalDeviceCooperativeMatrixPropertiesKHR::default();
    let mut properties = vk::PhysicalDeviceProperties2::default()
        .push_next(&mut subgroup)
        .push_next(&mut matrix);
    unsafe {
        instance
            .instance
            .get_physical_device_properties2(physical, &mut properties)
    };
    let limits = &properties.properties.limits;
    let enough_resources = limits.max_compute_shared_memory_size >= 16_640
        && limits.max_per_stage_descriptor_storage_buffers >= 5
        && limits.max_descriptor_set_storage_buffers >= 5;
    if !matrix
        .cooperative_matrix_supported_stages
        .contains(vk::ShaderStageFlags::COMPUTE)
        || !subgroup
            .supported_stages
            .contains(vk::ShaderStageFlags::COMPUTE)
        || !subgroup
            .supported_operations
            .contains(vk::SubgroupFeatureFlags::VOTE)
        || !matches!(subgroup.subgroup_size, 32 | 64)
        || !enough_resources
    {
        return Ok(None);
    }
    let loader = ash::khr::cooperative_matrix::Instance::new(&instance._entry, &instance.instance);
    let matrices = checked("enumerate cooperative matrix properties", unsafe {
        loader.get_physical_device_cooperative_matrix_properties(physical)
    })?;
    Ok(matrices
        .iter()
        .any(supports_pointwise_matrix)
        .then_some(subgroup.subgroup_size))
}

fn supports_pointwise_matrix(matrix: &vk::CooperativeMatrixPropertiesKHR<'_>) -> bool {
    matrix.m_size == 16
        && matrix.n_size == 16
        && matrix.k_size == 16
        && matrix.a_type == vk::ComponentTypeKHR::FLOAT16
        && matrix.b_type == vk::ComponentTypeKHR::FLOAT16
        && matrix.c_type == vk::ComponentTypeKHR::FLOAT32
        && matrix.result_type == vk::ComponentTypeKHR::FLOAT32
        && matrix.saturating_accumulation == vk::FALSE
        && matrix.scope == vk::ScopeKHR::SUBGROUP
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

fn select_memory_type(
    memory: &vk::PhysicalDeviceMemoryProperties,
    compatible: u32,
    required: vk::MemoryPropertyFlags,
    preferred: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let mut fallback = None;
    for index in 0..memory.memory_type_count {
        let flags = memory.memory_types[index as usize].property_flags;
        if compatible & (1 << index) == 0 || !flags.contains(required) {
            continue;
        }
        if flags.contains(preferred) {
            return Some(index);
        }
        fallback.get_or_insert(index);
    }
    fallback
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
        // Readback benefits from CPU caching. Keep HOST_COHERENT mandatory and
        // retain the original compatible type if cached memory is unavailable.
        let preferred = if properties.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            && usage.contains(vk::BufferUsageFlags::TRANSFER_DST)
        {
            vk::MemoryPropertyFlags::HOST_CACHED
        } else {
            vk::MemoryPropertyFlags::empty()
        };
        let memory_type = select_memory_type(
            &context.memory,
            requirements.memory_type_bits,
            properties,
            preferred,
        )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_readback_memory_preserves_coherence_and_compatible_type_bits() {
        let coherent =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let cached = vk::MemoryPropertyFlags::HOST_CACHED;
        let mut memory = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 3,
            ..Default::default()
        };
        memory.memory_types[0].property_flags = coherent;
        memory.memory_types[1].property_flags = coherent | cached;
        memory.memory_types[2].property_flags = vk::MemoryPropertyFlags::HOST_VISIBLE | cached;
        assert_eq!(
            select_memory_type(&memory, 0b111, coherent, cached),
            Some(1)
        );
        // Cached but non-coherent memory must not displace a valid fallback.
        assert_eq!(
            select_memory_type(&memory, 0b101, coherent, cached),
            Some(0)
        );
        assert_eq!(select_memory_type(&memory, 0b100, coherent, cached), None);
        assert_eq!(
            select_memory_type(&memory, 0b111, coherent, vk::MemoryPropertyFlags::empty()),
            Some(0)
        );
    }

    #[test]
    fn incompatible_matrix_types_and_shapes_are_not_enabled() {
        let supported = vk::CooperativeMatrixPropertiesKHR::default()
            .m_size(16)
            .n_size(16)
            .k_size(16)
            .a_type(vk::ComponentTypeKHR::FLOAT16)
            .b_type(vk::ComponentTypeKHR::FLOAT16)
            .c_type(vk::ComponentTypeKHR::FLOAT32)
            .result_type(vk::ComponentTypeKHR::FLOAT32)
            .scope(vk::ScopeKHR::SUBGROUP);
        assert!(supports_pointwise_matrix(&supported));
        for incompatible in [
            supported.m_size(8),
            supported.n_size(8),
            supported.k_size(8),
            supported.a_type(vk::ComponentTypeKHR::FLOAT32),
            supported.b_type(vk::ComponentTypeKHR::FLOAT32),
            supported.c_type(vk::ComponentTypeKHR::FLOAT16),
            supported.result_type(vk::ComponentTypeKHR::FLOAT16),
            supported.scope(vk::ScopeKHR::WORKGROUP),
            supported.saturating_accumulation(true),
        ] {
            assert!(!supports_pointwise_matrix(&incompatible));
        }
    }
}
