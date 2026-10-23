use super::{Gpu, Resource, checked, memory_barrier};
use anr_core::Result;
use ash::vk;

const SAMPLES: usize = 5;
const REPEATS: u32 = 32;

// Ties retain the first (portable/default) implementation. A single scheduling
// spike must not turn an otherwise slower implementation into the winner.
pub(super) fn fastest(samples: &mut [[u64; SAMPLES]]) -> usize {
    for sample in samples.iter_mut() {
        sample.sort_unstable();
    }
    samples
        .iter()
        .enumerate()
        .min_by_key(|(_, sample)| sample[SAMPLES / 2])
        .map(|(index, _)| index)
        .expect("kernel candidates are nonempty")
}

pub(super) struct Tuner {
    queries: Resource<vk::QueryPool>,
    _pool: Resource<vk::CommandPool>,
    commands: Vec<vk::CommandBuffer>,
    mask: u64,
}

impl Tuner {
    pub(super) fn new(gpu: &Gpu) -> Result<Option<Self>> {
        let bits = gpu.context.timestamp_valid_bits;
        if bits == 0 {
            return Ok(None);
        }
        let device = &gpu.context.device;
        let queries = Resource::new(
            device,
            checked("create tuning queries", unsafe {
                device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(2),
                    None,
                )
            })?,
            |d, h| unsafe { d.destroy_query_pool(h, None) },
        );
        let pool = Resource::new(
            device,
            checked("create tuning command pool", unsafe {
                device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(gpu.context.family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
            })?,
            |d, h| unsafe { d.destroy_command_pool(h, None) },
        );
        let commands = checked("allocate tuning commands", unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool.value)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(3),
            )
        })?;
        let mask = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        Ok(Some(Self {
            queries,
            _pool: pool,
            commands,
            mask,
        }))
    }

    pub(super) fn choose(
        &mut self,
        gpu: &mut Gpu,
        index: usize,
        candidates: &[(vk::Pipeline, u32)],
    ) -> Result<usize> {
        assert!(!candidates.is_empty() && candidates.len() <= self.commands.len());
        let device = gpu.context.device.clone();
        for (candidate, &(pipeline, groups)) in candidates.iter().enumerate() {
            let command = self.commands[candidate];
            checked("reset tuning commands", unsafe {
                device.reset_command_buffer(command, vk::CommandBufferResetFlags::empty())
            })?;
            checked("begin tuning commands", unsafe {
                device.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
            })?;
            unsafe {
                device.cmd_reset_query_pool(command, self.queries.value, 0, 2);
                memory_barrier(
                    &device,
                    command,
                    vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::SHADER_WRITE
                        | vk::AccessFlags::TRANSFER_WRITE
                        | vk::AccessFlags::HOST_WRITE,
                    vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::SHADER_WRITE
                        | vk::AccessFlags::UNIFORM_READ,
                );
                device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    gpu.pipeline_layout.value,
                    0,
                    &[gpu.descriptor_sets[0]],
                    &[gpu.dispatches[index].parameter_index as u32 * gpu.parameter_stride],
                );
                device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    self.queries.value,
                    0,
                );
                for _ in 0..REPEATS {
                    device.cmd_dispatch(command, groups, 1, 1);
                    memory_barrier(
                        &device,
                        command,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::AccessFlags::SHADER_WRITE,
                        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                    );
                }
                device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.queries.value,
                    1,
                );
            }
            checked("end tuning commands", unsafe {
                device.end_command_buffer(command)
            })?;
            // First execution warms shader/weight caches; it is not scored.
            gpu.submit_and_wait(command)?;
        }
        let mut samples = [[0; SAMPLES]; 3];
        for sample in 0..SAMPLES {
            // Alternate order to avoid consistently giving one candidate the
            // first submission while the device changes clock or workload.
            let mut order = samples[..candidates.len()].iter_mut().enumerate();
            while let Some((candidate, timings)) = if sample % 2 == 0 {
                order.next()
            } else {
                order.next_back()
            } {
                gpu.submit_and_wait(self.commands[candidate])?;
                let mut timestamps = [0u64; 2];
                checked("read tuning timestamps", unsafe {
                    device.get_query_pool_results(
                        self.queries.value,
                        0,
                        &mut timestamps,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                })?;
                timings[sample] = timestamps[1].wrapping_sub(timestamps[0]) & self.mask;
            }
        }
        Ok(fastest(&mut samples[..candidates.len()]))
    }

    // Tuning can write scratch and the alternate recurrent-state buffer. Never
    // let those synthetic executions become the first frame's business state.
    pub(super) fn clear(&mut self, gpu: &mut Gpu) -> Result<()> {
        let device = gpu.context.device.clone();
        let command = self.commands[0];
        checked("reset post-tuning commands", unsafe {
            device.reset_command_buffer(command, vk::CommandBufferResetFlags::empty())
        })?;
        checked("begin post-tuning commands", unsafe {
            device.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
        })?;
        unsafe {
            memory_barrier(
                &device,
                command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::TRANSFER_WRITE,
            );
            for state in &gpu.buffers.states {
                device.cmd_fill_buffer(command, state.handle, 0, vk::WHOLE_SIZE, 0);
            }
            device.cmd_fill_buffer(command, gpu.buffers.scratch.handle, 0, vk::WHOLE_SIZE, 0);
            memory_barrier(
                &device,
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_WRITE,
            );
        }
        checked("end post-tuning commands", unsafe {
            device.end_command_buffer(command)
        })?;
        gpu.submit_and_wait(command)
    }
}
