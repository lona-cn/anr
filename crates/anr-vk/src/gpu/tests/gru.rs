use super::*;

// This catches wrong weight transposition, repeated bias per K lane, a reset
// applied before Rh's bias, missing reductions, and omitted state_out writes.
// The oracle reads the original [k][six gates][j] weights, never prepare()'s
// layout or the shader's parallel reduction order.
fn oracle(weights: &[f32], p: &[u32; 32], input: &[f32; 256], old: &[f32; 256]) -> [f64; 256] {
    std::array::from_fn(|j| {
        let mut dot: [f64; 6] =
            std::array::from_fn(|gate| f64::from(weights[p[10] as usize + gate * 256 + j]));
        for k in 0..256 {
            for (gate, sum) in dot.iter_mut().enumerate() {
                let value = if gate < 3 { input[k] } else { old[k] };
                *sum += f64::from(value)
                    * f64::from(weights[p[4] as usize + k * 1536 + gate * 256 + j]);
            }
        }
        let z = 1.0 / (1.0 + (-(dot[0] + dot[3])).exp());
        let r = 1.0 / (1.0 + (-(dot[1] + dot[4])).exp());
        let candidate = (dot[2] + r * dot[5]).tanh();
        (1.0 - z) * candidate + z * f64::from(old[j])
    })
}

// Adjacent K pairs cancel in a selected real model row. Both multiplicands
// stay O(1), rather than saturating every gate with huge adversarial values.
fn cancelling_vector(weights: &[f32], p: &[u32; 32], gate: usize) -> [f32; 256] {
    let mut values = [0.0; 256];
    let mut sum = 0.0_f64;
    let mut magnitude = 0.0_f64;
    for pair in 0..128 {
        let k = pair * 2;
        let a = weights[p[4] as usize + k * 1536 + gate * 256 + 113];
        let b = weights[p[4] as usize + (k + 1) * 1536 + gate * 256 + 113];
        let scale = a.abs().max(b.abs());
        if scale != 0.0 {
            values[k] = 2.0 * (b / scale);
            values[k + 1] = -2.0 * (a / scale);
        } else {
            values[k] = 0.75;
            values[k + 1] = -0.5;
        }
        let first = f64::from(a) * f64::from(values[k]);
        let second = f64::from(b) * f64::from(values[k + 1]);
        sum += first + second;
        magnitude += first.abs() + second.abs();
    }
    assert!(magnitude > 1.0e-3, "model row must exercise cancellation");
    assert!(sum.abs() <= magnitude * 1.0e-6);
    values
}

fn run_gru(
    gpu: &mut Gpu,
    index: usize,
    input: &[f32; 256],
    old: &[f32; 256],
) -> ([f32; 256], [f32; 256]) {
    let p = gpu.model.dispatches[index].params;
    let execution = &gpu.dispatches[index];
    let upload = Buffer::new(
        &gpu.context,
        2048,
        vk::BufferUsageFlags::TRANSFER_SRC,
        HOST_MEMORY,
    )
    .unwrap();
    let readback = Buffer::new(
        &gpu.context,
        2048,
        vk::BufferUsageFlags::TRANSFER_DST,
        HOST_MEMORY,
    )
    .unwrap();
    unsafe {
        ptr::copy_nonoverlapping(input.as_ptr().cast::<u8>(), upload.mapped, 1024);
        ptr::copy_nonoverlapping(old.as_ptr().cast::<u8>(), upload.mapped.add(1024), 1024);
    }
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
            vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::MEMORY_READ
                | vk::AccessFlags::MEMORY_WRITE
                | vk::AccessFlags::HOST_WRITE,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
        );
        device.cmd_copy_buffer(
            command,
            upload.handle,
            gpu.buffers.scratch.handle,
            &[vk::BufferCopy::default()
                .dst_offset(u64::from(p[2]) * 4)
                .size(1024)],
        );
        device.cmd_copy_buffer(
            command,
            upload.handle,
            gpu.buffers.states[0].handle,
            &[vk::BufferCopy::default()
                .src_offset(1024)
                .dst_offset(u64::from(p[8]) * 4)
                .size(1024)],
        );
        memory_barrier(
            device,
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        device.cmd_bind_pipeline(
            command,
            vk::PipelineBindPoint::COMPUTE,
            gpu.pipelines[execution.pipeline].value,
        );
        device.cmd_bind_descriptor_sets(
            command,
            vk::PipelineBindPoint::COMPUTE,
            gpu.pipeline_layout.value,
            0,
            &[gpu.descriptor_sets[0]],
            &[execution.parameter_index as u32 * gpu.parameter_stride],
        );
        device.cmd_dispatch(command, execution.groups, 1, 1);
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
                .src_offset(u64::from(p[3]) * 4)
                .size(1024)],
        );
        device.cmd_copy_buffer(
            command,
            gpu.buffers.states[1].handle,
            readback.handle,
            &[vk::BufferCopy::default()
                .src_offset(u64::from(p[8]) * 4)
                .dst_offset(1024)
                .size(1024)],
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
    let mut output = [0.0; 256];
    let mut state = [0.0; 256];
    unsafe {
        ptr::copy_nonoverlapping(readback.mapped.cast::<f32>(), output.as_mut_ptr(), 256);
        ptr::copy_nonoverlapping(
            readback.mapped.add(1024).cast::<f32>(),
            state.as_mut_ptr(),
            256,
        );
    }
    (output, state)
}

#[test]
#[ignore = "requires ANR_DFN3_MODEL and a Vulkan 1.1 compute device"]
fn parallel_gru_matches_original_weights_with_nonzero_state_and_cancellation() {
    let path = std::env::var_os("ANR_DFN3_MODEL").expect("set ANR_DFN3_MODEL");
    let model = Model::load(std::path::Path::new(&path)).unwrap();
    let weights = model.weights.clone();
    let indices: Vec<_> = model
        .dispatches
        .iter()
        .enumerate()
        .filter_map(|(index, dispatch)| {
            (dispatch.shader == ShaderKind::Network && dispatch.params[0] == 5).then_some(index)
        })
        .collect();
    assert_eq!(indices.len(), 5);
    let mut gpu = Gpu::new(model).unwrap();
    for width in [16, 32, 64] {
        for &index in &indices {
            let p = gpu.model.dispatches[index].params;
            let words = shaders::network_spirv(&p, shaders::NetworkKernel::Gru(width)).unwrap();
            let pipeline = pipeline_words(
                &gpu.context.device,
                gpu.pipeline_layout.value,
                &words,
                "GRU oracle",
                vk::PipelineShaderStageCreateFlags::empty(),
            )
            .unwrap();
            let pipeline_index = gpu.dispatches[index].pipeline;
            // Every preceding run waited for completion. These test executors
            // never submit the full-frame command buffers recorded by Gpu::new.
            drop(mem::replace(&mut gpu.pipelines[pipeline_index], pipeline));
            gpu.dispatches[index].groups = super::super::gru::groups(width);
            for cancellation in [false, true] {
                let (input, old) = if cancellation {
                    (
                        cancelling_vector(&weights, &p, 0),
                        cancelling_vector(&weights, &p, 3),
                    )
                } else {
                    (
                        std::array::from_fn(|k| ((k * 37 + 11) % 257) as f32 / 127.0 - 0.97),
                        std::array::from_fn(|k| ((k * 53 + 29) % 251) as f32 / 157.0 - 0.79),
                    )
                };
                let expected = oracle(&weights, &p, &input, &old);
                let (output, state) = run_gru(&mut gpu, index, &input, &old);
                let mut maximum = 0.0_f64;
                let mut squared = 0.0_f64;
                for j in 0..256 {
                    assert!(output[j].is_finite());
                    assert_eq!(
                        output[j].to_bits(),
                        state[j].to_bits(),
                        "GRU {index}, width {width}, cancellation {cancellation}, neuron {j}: scratch/state mismatch"
                    );
                    let error = (f64::from(output[j]) - expected[j]).abs();
                    maximum = maximum.max(error);
                    squared += error * error;
                }
                let rms = (squared / 256.0).sqrt();
                assert!(
                    maximum <= 2.0e-5 && rms <= 2.0e-6,
                    "GRU {index}, width {width}, cancellation {cancellation}: maximum={maximum:e}, RMS={rms:e} against independent f64 oracle"
                );
            }
        }
    }
}
