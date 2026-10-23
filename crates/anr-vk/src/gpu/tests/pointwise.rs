use super::*;

fn run_operator(gpu: &mut Gpu, index: usize, input: &[f32]) -> Vec<f32> {
    let dispatch = &gpu.model.dispatches[index];
    let output_len = dispatch.params[1] as usize;
    let upload = Buffer::new(
        &gpu.context,
        input.len() as u64 * 4,
        vk::BufferUsageFlags::TRANSFER_SRC,
        HOST_MEMORY,
    )
    .unwrap();
    let readback = Buffer::new(
        &gpu.context,
        output_len as u64 * 4,
        vk::BufferUsageFlags::TRANSFER_DST,
        HOST_MEMORY,
    )
    .unwrap();
    unsafe {
        ptr::copy_nonoverlapping(input.as_ptr().cast::<u8>(), upload.mapped, input.len() * 4)
    };
    let device = &gpu.context.device;
    let pool = Resource::new(
        device,
        unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(gpu.context.family),
                None,
            )
        }
        .unwrap(),
        |device, pool| unsafe { device.destroy_command_pool(pool, None) },
    );
    let command = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool.value)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .unwrap()[0];
    unsafe {
        device
            .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
            .unwrap();
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
            upload.handle,
            gpu.buffers.scratch.handle,
            &[vk::BufferCopy::default()
                .dst_offset(u64::from(dispatch.params[2]) * 4)
                .size(input.len() as u64 * 4)],
        );
        memory_barrier(
            device,
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        let execution = gpu.dispatches[index];
        let pipeline = gpu.pipelines[execution.pipeline].value;
        let groups = execution.groups;
        device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            command,
            vk::PipelineBindPoint::COMPUTE,
            gpu.pipeline_layout.value,
            0,
            &[gpu.descriptor_sets[0]],
            &[execution.parameter_index as u32 * gpu.parameter_stride],
        );
        device.cmd_dispatch(command, groups, 1, 1);
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
            gpu.buffers.scratch.handle,
            readback.handle,
            &[vk::BufferCopy::default()
                .src_offset(u64::from(dispatch.params[3]) * 4)
                .size(output_len as u64 * 4)],
        );
        memory_barrier(
            device,
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::HOST_READ,
        );
        device.end_command_buffer(command).unwrap();
    }
    gpu.submit_and_wait(command).unwrap();
    let mut output = vec![0.0; output_len];
    unsafe {
        ptr::copy_nonoverlapping(
            readback.mapped.cast::<f32>(),
            output.as_mut_ptr(),
            output_len,
        )
    };
    output
}

#[test]
#[ignore = "requires ANR_DFN3_MODEL and a compatible KHR cooperative-matrix device"]
fn pointwise_tiles_match_fp32_with_cancellation_and_spatial_tails() {
    let path = std::env::var_os("ANR_DFN3_MODEL").expect("set ANR_DFN3_MODEL");
    let weights = Model::load(std::path::Path::new(&path)).unwrap().weights;
    let mut gpu = executor(true);
    assert!(gpu.context.cooperative_subgroup_size.is_some());
    let mut scalar_gpu = executor(false);
    let indices: Vec<_> = gpu
        .model
        .dispatches
        .iter()
        .enumerate()
        .filter_map(|(index, dispatch)| pointwise_groups(dispatch, 32).map(|_| index))
        .collect();
    for index in indices {
        let p = gpu.model.dispatches[index].params;
        let spatial = (p[10] * p[11]) as usize;
        for case in [
            "ordinary",
            "half-boundaries",
            "last-half-flag",
            "mixed-fallback",
        ] {
            let mut input: Vec<f32> = (0..64 * spatial)
                .map(|i| {
                    let value = ((i * 37 + 11) % 257) as f32 / 127.0 - 1.0;
                    value + (i % 7) as f32 * 0.000031
                })
                .collect();
            match case {
                "half-boundaries" => input[..6].copy_from_slice(&[
                    32_784.0,
                    -32_784.0,
                    65_504.0,
                    -65_504.0,
                    2.0_f32.powi(-24),
                    -2.0_f32.powi(-24),
                ]),
                // This value is loaded only by local invocation 32. Both SG32s
                // must inspect the second half of the whole-workgroup flags.
                "last-half-flag" => input[2 * spatial] = 70_000.0,
                "mixed-fallback" if spatial >= 32 => {
                    input[0] = 70_000.0;
                    input[1] = -90_000.0;
                    input[16] = 32_784.0;
                    input[17] = -32_784.0;
                }
                _ => {}
            }
            let scalar = run_operator(&mut scalar_gpu, index, &input);
            let cooperative = run_operator(&mut gpu, index, &input);
            let mut maximum = 0.0_f64;
            let mut scalar_maximum = 0.0_f64;
            for channel in 0..64 {
                for position in 0..spatial {
                    let bias = f64::from(weights[p[5] as usize + channel]);
                    let mut expected = bias;
                    let mut absolute_sum = bias.abs();
                    for k in 0..64 {
                        let product = f64::from(input[k * spatial + position])
                            * f64::from(weights[p[4] as usize + channel * 64 + k]);
                        expected += product;
                        absolute_sum += product.abs();
                    }
                    expected = expected.max(0.0);
                    let output_index = channel * spatial + position;
                    let scalar_error = (f64::from(scalar[output_index]) - expected).abs();
                    let error = (f64::from(cooperative[output_index]) - expected).abs();
                    // Standard forward-error bound for 64 FP32 products plus bias.
                    // A few ULPs of the final result do not bound cancellation or
                    // rounding in the original sequential FP32 accumulation.
                    let unit_roundoff = f64::from(f32::EPSILON) / 2.0;
                    let gamma = 65.0 * unit_roundoff / (1.0 - 65.0 * unit_roundoff);
                    let tolerance = gamma * absolute_sum + 1.0e-7;
                    assert!(
                        error <= tolerance,
                        "pointwise {index}, {case}, channel {channel}, position {position}: cooperative error {error:e}, original scalar error {scalar_error:e}, FP32 error bound {tolerance:e}"
                    );
                    assert!(scalar_error <= tolerance);
                    maximum = maximum.max(error);
                    scalar_maximum = scalar_maximum.max(scalar_error);
                }
            }
            eprintln!(
                "pointwise index={index}, spatial={spatial}, case={case}: scalar/double={scalar_maximum:e}, cooperative/double={maximum:e}"
            );
        }
    }
}
