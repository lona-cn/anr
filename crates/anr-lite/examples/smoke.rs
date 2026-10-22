use std::{error::Error, time::Instant};

use anr_lite::{AUDIO_DELAY_SAMPLES, AnrLite48, AudioProcessor, FRAME_SIZE, SAMPLE_RATE};
use burn::{
    module::Module,
    tensor::{Device, Tensor, TensorData},
};

fn main() -> Result<(), Box<dyn Error>> {
    let gpu = match std::env::args().nth(1).as_deref() {
        None => false,
        Some("--wgpu") => true,
        Some(_) => return Err(
            "用法：cargo run -p anr-lite --release --example smoke [--features wgpu] -- [--wgpu]"
                .into(),
        ),
    };
    let device = if gpu {
        #[cfg(feature = "wgpu")]
        {
            Device::wgpu(burn::tensor::DeviceKind::DiscreteGpu(0))
        }
        #[cfg(not(feature = "wgpu"))]
        {
            return Err("wgpu 未编译；请添加 --features wgpu".into());
        }
    } else {
        Device::flex()
    };
    device.seed(27);
    let model = AnrLite48::new(&device).valid();
    let processor = AudioProcessor::new(&device);
    let samples = 6 * FRAME_SIZE + 137;
    let input: Vec<f32> = (0..samples)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            0.15 * (std::f32::consts::TAU * 317.0 * t).sin()
                + 0.04 * (std::f32::consts::TAU * 1259.0 * t).cos()
        })
        .collect();
    let started = Instant::now();
    let whole = processor
        .forward(
            &model,
            Tensor::<2>::from_data(TensorData::new(input.clone(), [1, samples]), &device),
        )
        .into_data()
        .try_to_vec::<f32>()?;
    let elapsed = started.elapsed();
    let mut state = processor.initial_state(1);
    let mut streaming = Vec::with_capacity(samples + AUDIO_DELAY_SAMPLES + FRAME_SIZE);
    for chunk in input.chunks(FRAME_SIZE) {
        let mut frame = vec![0.0; FRAME_SIZE];
        frame[..chunk.len()].copy_from_slice(chunk);
        let output = processor.forward_frame(
            &model,
            Tensor::<2>::from_data(TensorData::new(frame, [1, FRAME_SIZE]), &device),
            &mut state,
        );
        streaming.extend(output.into_data().try_to_vec::<f32>()?);
    }
    streaming.extend(
        processor
            .flush(&model, &mut state)
            .into_data()
            .try_to_vec::<f32>()?,
    );
    let aligned = &streaming[AUDIO_DELAY_SAMPLES..AUDIO_DELAY_SAMPLES + samples];
    let max_difference = whole
        .iter()
        .zip(aligned)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    if whole.len() != samples || whole.iter().any(|v| !v.is_finite()) || max_difference > 3e-4 {
        return Err(format!(
            "模型执行验证失败：输出长度={}，最大流式差异={max_difference}",
            whole.len()
        )
        .into());
    }
    // Backward is a correctness check only: no optimizer is created and no
    // parameter update is performed.
    let gradient_device = device.clone().autodiff();
    let gradient_model = model.clone().train();
    let gradient_processor = AudioProcessor::new(&gradient_device);
    let gradient_input = Tensor::<2>::from_data(
        TensorData::new(input[..2 * FRAME_SIZE].to_vec(), [1, 2 * FRAME_SIZE]),
        &gradient_device,
    )
    .require_grad();
    let prediction = gradient_processor.forward(&gradient_model, gradient_input.clone());
    let gradients = (prediction.clone() * prediction).mean().backward();
    let input_gradient = gradient_input
        .grad(&gradients)
        .ok_or("音频输入未收到梯度")?
        .try_into_vec_as::<f32>()?;
    if input_gradient.iter().any(|value| !value.is_finite())
        || !input_gradient.iter().any(|value| value.abs() > 1e-9)
    {
        return Err("音频输入梯度无效".into());
    }
    println!("ANR-Lite-48：随机初始化，尚未训练；本示例不验证降噪效果。");
    println!("执行设备：{device:?}");
    println!("Burn Module 参数统计：{}", model.num_params());
    println!("输入/输出：{samples} samples，{SAMPLE_RATE} Hz，单声道");
    println!("流式采样延迟：{AUDIO_DELAY_SAMPLES} samples（不含当前输入帧的收集时间）");
    println!("整段处理与逐帧加 flush 的最大差异：{max_difference:.8}");
    println!("输入梯度：有限且非零；未执行优化器更新。");
    println!("本次整段执行含首次调用开销：{elapsed:?}（不是稳态性能基准）");
    Ok(())
}
