mod native;

use anr_core::Result;
use anr_dfn3::{
    Executor, FRAME_SIZE, Model, PARAMETER_BYTES, STATE_SIZE, ShaderKind, backend, shaders,
};
use ash::vk;
use native::{Buffer, Context, Resource, checked};
use std::{mem, ptr};

const FRAME_BYTES: u64 = FRAME_SIZE as u64 * 4;
const STATE_BYTES: u64 = STATE_SIZE as u64 * 4;
const HOST_MEMORY: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
    vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
        | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
);

struct Buffers {
    weights: Buffer,
    states: [Buffer; 2],
    scratch: Buffer,
    parameters: Buffer,
    upload: Buffer,
    readback: Buffer,
}

// Only the worker ever touches these handles or mapped pointers. Resources
// precede Context so Rust's field drop order always keeps the device alive.
// Four immutable frame command buffers cover both state directions and reset;
// submission is the only driver command construction on the frame hot path.
pub(crate) struct Gpu {
    model: Model,
    current: usize,
    poisoned: bool,
    parameter_stride: u32,
    buffers: Buffers,
    pipelines: [Resource<vk::Pipeline>; 3],
    pipeline_layout: Resource<vk::PipelineLayout>,
    _descriptor_pool: Resource<vk::DescriptorPool>,
    _descriptor_layout: Resource<vk::DescriptorSetLayout>,
    _command_pool: Resource<vk::CommandPool>,
    fence: Resource<vk::Fence>,
    descriptor_sets: [vk::DescriptorSet; 2],
    commands: [vk::CommandBuffer; 5],
    context: Context,
}

impl Gpu {
    pub(crate) fn new(model: Model) -> Result<Self> {
        let context = Context::new(&model)?;
        let alignment = context.limits.min_uniform_buffer_offset_alignment.max(1);
        let stride = u64::from(PARAMETER_BYTES)
            .div_ceil(alignment)
            .checked_mul(alignment)
            .ok_or_else(|| backend("Vulkan dynamic uniform alignment overflows"))?;
        let parameter_stride = u32::try_from(stride)
            .map_err(|_| backend("Vulkan dynamic uniform stride exceeds u32"))?;
        let parameter_size = stride
            .checked_mul(model.dispatches.len() as u64)
            .ok_or_else(|| backend("Vulkan parameter buffer size overflows"))?;
        if model.dispatches.is_empty()
            || parameter_size - stride > u64::from(u32::MAX)
            || parameter_size > usize::MAX as u64
        {
            return Err(backend(
                "DFN3 dispatch parameters exceed Vulkan dynamic offset range",
            ));
        }
        let storage_usage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let buffers = Buffers {
            weights: Buffer::new(
                &context,
                model.weights.len() as u64 * 4,
                storage_usage,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?,
            states: [
                Buffer::new(
                    &context,
                    STATE_BYTES,
                    storage_usage,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )?,
                Buffer::new(
                    &context,
                    STATE_BYTES,
                    storage_usage,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )?,
            ],
            scratch: Buffer::new(
                &context,
                u64::from(model.scratch_len) * 4,
                storage_usage | vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?,
            parameters: Buffer::new(
                &context,
                parameter_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                HOST_MEMORY,
            )?,
            upload: Buffer::new(
                &context,
                FRAME_BYTES,
                vk::BufferUsageFlags::TRANSFER_SRC,
                HOST_MEMORY,
            )?,
            readback: Buffer::new(
                &context,
                FRAME_BYTES,
                vk::BufferUsageFlags::TRANSFER_DST,
                HOST_MEMORY,
            )?,
        };
        let device = &context.device;
        let bindings = [
            storage_binding(0),
            storage_binding(1),
            storage_binding(2),
            storage_binding(3),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let descriptor_layout = Resource::new(
            device,
            checked("create descriptor layout", unsafe {
                device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
            })?,
            |d, h| unsafe { d.destroy_descriptor_set_layout(h, None) },
        );
        let layouts = [descriptor_layout.value];
        let pipeline_layout = Resource::new(
            device,
            checked("create pipeline layout", unsafe {
                device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                    None,
                )
            })?,
            |d, h| unsafe { d.destroy_pipeline_layout(h, None) },
        );
        let pipelines = [
            pipeline(device, pipeline_layout.value, ShaderKind::Network)?,
            pipeline(device, pipeline_layout.value, ShaderKind::Dsp)?,
            pipeline(device, pipeline_layout.value, ShaderKind::Fft)?,
        ];
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(8),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(2),
        ];
        let descriptor_pool = Resource::new(
            device,
            checked("create descriptor pool", unsafe {
                device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(2)
                        .pool_sizes(&pool_sizes),
                    None,
                )
            })?,
            |d, h| unsafe { d.destroy_descriptor_pool(h, None) },
        );
        let set_layouts = [descriptor_layout.value; 2];
        let allocated = checked("allocate descriptor sets", unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool.value)
                    .set_layouts(&set_layouts),
            )
        })?;
        let descriptor_sets = [allocated[0], allocated[1]];
        for (current, &set) in descriptor_sets.iter().enumerate() {
            let infos = [
                descriptor_buffer(&buffers.weights),
                descriptor_buffer(&buffers.states[current]),
                descriptor_buffer(&buffers.states[current ^ 1]),
                descriptor_buffer(&buffers.scratch),
                vk::DescriptorBufferInfo::default()
                    .buffer(buffers.parameters.handle)
                    .offset(0)
                    .range(u64::from(PARAMETER_BYTES)),
            ];
            let writes = [
                descriptor_write(set, 0, vk::DescriptorType::STORAGE_BUFFER, &infos[0]),
                descriptor_write(set, 1, vk::DescriptorType::STORAGE_BUFFER, &infos[1]),
                descriptor_write(set, 2, vk::DescriptorType::STORAGE_BUFFER, &infos[2]),
                descriptor_write(set, 3, vk::DescriptorType::STORAGE_BUFFER, &infos[3]),
                descriptor_write(
                    set,
                    4,
                    vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    &infos[4],
                ),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        let command_pool = Resource::new(
            device,
            checked("create command pool", unsafe {
                device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(context.family),
                    None,
                )
            })?,
            |d, h| unsafe { d.destroy_command_pool(h, None) },
        );
        let allocated = checked("allocate command buffers", unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool.value)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(5),
            )
        })?;
        let commands = [
            allocated[0],
            allocated[1],
            allocated[2],
            allocated[3],
            allocated[4],
        ];
        let fence = Resource::new(
            device,
            checked("create completion fence", unsafe {
                device.create_fence(&vk::FenceCreateInfo::default(), None)
            })?,
            |d, h| unsafe { d.destroy_fence(h, None) },
        );
        let mut gpu = Self {
            model,
            current: 0,
            poisoned: false,
            parameter_stride,
            buffers,
            pipelines,
            pipeline_layout,
            _descriptor_pool: descriptor_pool,
            _descriptor_layout: descriptor_layout,
            _command_pool: command_pool,
            fence,
            descriptor_sets,
            commands,
            context,
        };
        gpu.initialize()?;
        for current in 0..2 {
            gpu.record_frame(gpu.commands[current * 2], current, false)?;
            gpu.record_frame(gpu.commands[current * 2 + 1], current, true)?;
        }
        Ok(gpu)
    }
}
impl Executor for Gpu {
    fn frame(
        &mut self,
        input: &[f32; FRAME_SIZE],
        output: &mut [f32; FRAME_SIZE],
        attenuation: f32,
        reset: bool,
    ) -> Result<()> {
        if self.poisoned {
            return Err(backend(
                "Vulkan execution failed; recreate the denoiser before reuse",
            ));
        }
        // The previous call waited its fence, so these persistent coherent
        // mappings are no longer referenced by pending device work.
        unsafe {
            ptr::copy_nonoverlapping(
                input.as_ptr().cast::<u8>(),
                self.buffers.upload.mapped,
                FRAME_BYTES as usize,
            );
        }
        if self.model.update_attenuation(attenuation) {
            let index = self.model.attenuation_dispatch;
            let offset = index * self.parameter_stride as usize + 22 * 4;
            unsafe {
                ptr::copy_nonoverlapping(
                    self.model.dispatches[index].params[22..24]
                        .as_ptr()
                        .cast::<u8>(),
                    self.buffers.parameters.mapped.add(offset),
                    8,
                );
            }
        }
        let current = if reset { 0 } else { self.current };
        self.submit_and_wait(self.commands[current * 2 + usize::from(reset)])?;
        unsafe {
            ptr::copy_nonoverlapping(
                self.buffers.readback.mapped.cast::<f32>(),
                output.as_mut_ptr(),
                FRAME_SIZE,
            );
        }
        // Never advance state or expose output until successful completion.
        self.current = current ^ 1;
        Ok(())
    }
}

impl Gpu {
    fn initialize(&mut self) -> Result<()> {
        let staging = Buffer::new(
            &self.context,
            self.buffers.weights.size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            HOST_MEMORY,
        )?;
        let parameters = self.model.parameter_bytes(self.parameter_stride);
        unsafe {
            ptr::copy_nonoverlapping(
                self.model.weights.as_ptr().cast::<u8>(),
                staging.mapped,
                staging.size as usize,
            );
            ptr::copy_nonoverlapping(
                parameters.as_ptr(),
                self.buffers.parameters.mapped,
                parameters.len(),
            );
        }
        let device = &self.context.device;
        let command = self.commands[4];
        checked("begin initialization commands", unsafe {
            device.begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        })?;
        unsafe {
            memory_barrier(
                device,
                command,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::HOST_WRITE,
                vk::AccessFlags::TRANSFER_READ,
            );
            device.cmd_copy_buffer(
                command,
                staging.handle,
                self.buffers.weights.handle,
                &[vk::BufferCopy::default().size(staging.size)],
            );
            for state in &self.buffers.states {
                device.cmd_fill_buffer(command, state.handle, 0, vk::WHOLE_SIZE, 0);
            }
            device.cmd_fill_buffer(command, self.buffers.scratch.handle, 0, vk::WHOLE_SIZE, 0);
            memory_barrier(
                device,
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            );
        }
        checked("end initialization commands", unsafe {
            device.end_command_buffer(command)
        })?;
        // submit_and_wait drains pending work even on error, before staging
        // goes out of scope and its mapped allocation can be destroyed.
        self.submit_and_wait(command)?;
        drop(mem::take(&mut self.model.weights));
        Ok(())
    }

    fn record_frame(&self, command: vk::CommandBuffer, current: usize, reset: bool) -> Result<()> {
        let device = &self.context.device;
        checked("begin frame command buffer", unsafe {
            device.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
        })?;
        unsafe {
            // Cross-frame compute/transfer hazards, mapped upload writes and
            // mapped readback reads all precede this frame's first accesses.
            memory_barrier(
                device,
                command,
                vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE
                    | vk::AccessFlags::HOST_WRITE
                    | vk::AccessFlags::HOST_READ,
                vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE
                    | vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::UNIFORM_READ,
            );
            if reset {
                for state in &self.buffers.states {
                    device.cmd_fill_buffer(command, state.handle, 0, vk::WHOLE_SIZE, 0);
                }
            }
            device.cmd_copy_buffer(
                command,
                self.buffers.upload.handle,
                self.buffers.scratch.handle,
                &[vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(u64::from(self.model.input) * 4)
                    .size(FRAME_BYTES)],
            );
            memory_barrier(
                device,
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            );
            let mut last_shader = None;
            for (index, dispatch) in self.model.dispatches.iter().enumerate() {
                if last_shader != Some(dispatch.shader) {
                    device.cmd_bind_pipeline(
                        command,
                        vk::PipelineBindPoint::COMPUTE,
                        self.pipelines[dispatch.shader.index()].value,
                    );
                    last_shader = Some(dispatch.shader);
                }
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout.value,
                    0,
                    &[self.descriptor_sets[current]],
                    &[index as u32 * self.parameter_stride],
                );
                device.cmd_dispatch(command, dispatch.groups, 1, 1);
                // Every scheduled operator can consume scratch/state written by
                // its predecessor. No aliasing assumptions or CPU graph ops.
                memory_barrier(
                    device,
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::AccessFlags::SHADER_WRITE,
                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                );
            }
            memory_barrier(
                device,
                command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::TRANSFER_READ,
            );
            device.cmd_copy_buffer(
                command,
                self.buffers.scratch.handle,
                self.buffers.readback.handle,
                &[vk::BufferCopy::default()
                    .src_offset(u64::from(self.model.output) * 4)
                    .dst_offset(0)
                    .size(FRAME_BYTES)],
            );
            memory_barrier(
                device,
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::HOST_READ,
            );
        }
        checked("end frame command buffer", unsafe {
            device.end_command_buffer(command)
        })
    }

    fn submit_and_wait(&mut self, command: vk::CommandBuffer) -> Result<()> {
        let device = &self.context.device;
        let result = (|| {
            checked("reset completion fence", unsafe {
                device.reset_fences(&[self.fence.value])
            })?;
            let commands = [command];
            let submit = [vk::SubmitInfo::default().command_buffers(&commands)];
            checked("submit compute commands", unsafe {
                device.queue_submit(self.context.queue, &submit, self.fence.value)
            })?;
            checked("wait for compute completion", unsafe {
                device.wait_for_fences(&[self.fence.value], true, u64::MAX)
            })
        })();
        if result.is_err() {
            self.poisoned = true;
            // A failed submit/wait must not permit a mapped buffer update,
            // staging allocation release, or object destruction while in use.
            // Device loss also ends resource use; allocation failures do not.
            wait_idle_before_release(device);
        }
        result
    }
}

fn wait_idle_before_release(device: &ash::Device) {
    loop {
        match unsafe { device.device_wait_idle() } {
            Ok(()) | Err(vk::Result::ERROR_DEVICE_LOST) => return,
            // No safe timeout: retain ownership until completion is known,
            // even if an allocation failure makes teardown block.
            Err(_) => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // Run before any buffer, descriptor, pipeline, fence or pool field is
        // released, including failures after a queue submission in new().
        wait_idle_before_release(&self.context.device);
    }
}

fn storage_binding(binding: u32) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}
fn descriptor_buffer(buffer: &Buffer) -> vk::DescriptorBufferInfo {
    vk::DescriptorBufferInfo::default()
        .buffer(buffer.handle)
        .offset(0)
        .range(buffer.size)
}
fn descriptor_write<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    kind: vk::DescriptorType,
    info: &'a vk::DescriptorBufferInfo,
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(kind)
        .buffer_info(std::slice::from_ref(info))
}

fn pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    kind: ShaderKind,
) -> Result<Resource<vk::Pipeline>> {
    let words = shaders::spirv(kind)?;
    let module = Resource::new(
        device,
        checked("create compute shader module", unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        })?,
        |d, h| unsafe { d.destroy_shader_module(h, None) },
    );
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module.value)
        .name(c"main");
    let info = [vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout)];
    match unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &info, None) } {
        Ok(pipelines) => Ok(Resource::new(device, pipelines[0], |d, h| unsafe {
            d.destroy_pipeline(h, None)
        })),
        Err((pipelines, error)) => {
            // Vulkan may return successfully created handles even when a batch
            // reports failure. The one-pipeline case must still handle it.
            for pipeline in pipelines {
                if pipeline != vk::Pipeline::null() {
                    unsafe { device.destroy_pipeline(pipeline, None) };
                }
            }
            Err(backend(format!(
                "Vulkan create {kind:?} compute pipeline: {error:?}"
            )))
        }
    }
}

unsafe fn memory_barrier(
    device: &ash::Device,
    command: vk::CommandBuffer,
    source_stage: vk::PipelineStageFlags,
    destination_stage: vk::PipelineStageFlags,
    source_access: vk::AccessFlags,
    destination_access: vk::AccessFlags,
) {
    let barriers = [vk::MemoryBarrier::default()
        .src_access_mask(source_access)
        .dst_access_mask(destination_access)];
    unsafe {
        device.cmd_pipeline_barrier(
            command,
            source_stage,
            destination_stage,
            vk::DependencyFlags::empty(),
            &barriers,
            &[],
            &[],
        );
    }
}
