use crate::{
    args::{Args, Backend},
    statistics::Statistics,
};
use anr_core::NoiseReduction;
use anr_dfn3::{FRAME_SIZE, SAMPLE_RATE};
use anyhow::{Context, Result};
use serde::Serialize;
use std::{hint::black_box, time::Instant};

pub const INPUT_SEED: u32 = 0x1234_5678;

#[derive(Debug, Serialize)]
pub struct Measurement {
    pub initialization_ms: f64,
    pub effective_strength: f32,
    pub cpu_fallback_allowed: bool,
    #[serde(flatten)]
    pub statistics: Statistics,
}

#[derive(Default)]
pub struct Benchmark {
    #[cfg(feature = "ort")]
    ort_environment: Option<anr_ort::ort::environment::Environment>,
}

impl Benchmark {
    pub fn run(
        &mut self,
        backend: Backend,
        args: &Args,
        input: &[[f32; FRAME_SIZE]],
    ) -> Result<Measurement> {
        match backend {
            Backend::Gl => measure(args, input, false, || anr_gl::DFN3::from_file(&args.model)),
            Backend::Vk => measure(args, input, false, || anr_vk::DFN3::from_file(&args.model)),
            Backend::Dx => measure(args, input, false, || anr_dx::DFN3::from_file(&args.model)),
            Backend::OrtCuda | Backend::OrtTensorrt | Backend::OrtDirectml => {
                #[cfg(feature = "ort")]
                {
                    measure(args, input, true, || {
                        crate::ort_backend::load(backend, &args.model, &mut self.ort_environment)
                    })
                }
                #[cfg(not(feature = "ort"))]
                {
                    anyhow::bail!(
                        "后端 {} 未编译；请启用同名 Cargo feature，并选择 ort-download-binaries 或 ort-load-dynamic",
                        backend.name()
                    )
                }
            }
        }
    }
}

fn measure<D: NoiseReduction>(
    args: &Args,
    input: &[[f32; FRAME_SIZE]],
    cpu_fallback_allowed: bool,
    load: impl FnOnce() -> anr_core::Result<D>,
) -> Result<Measurement> {
    let started = Instant::now();
    let mut denoiser = load().context("加载模型或初始化 GPU 失败")?;
    let initialization_ms = started.elapsed().as_secs_f64() * 1_000.0;
    denoiser.set_factor(args.strength);
    let effective_strength = denoiser.factor();
    let mut output = [0.0_f32; FRAME_SIZE];
    let mut stream = input.iter().cycle();
    for (index, frame) in stream.by_ref().take(args.warmup as usize).enumerate() {
        denoiser
            .run(SAMPLE_RATE, frame, &mut output)
            .with_context(|| format!("第 {} 帧预热失败", index + 1))?;
        black_box(&output);
    }

    // Retain warmup state and input position. No reset or allocation in the timed loop.
    let mut latencies = Vec::with_capacity(args.frames as usize);
    let started = Instant::now();
    for (index, frame) in stream.take(args.frames as usize).enumerate() {
        let call_started = Instant::now();
        let result = denoiser.run(SAMPLE_RATE, frame, &mut output);
        let elapsed = call_started.elapsed();
        result.with_context(|| format!("第 {} 帧正式推理失败", index + 1))?;
        latencies.push(elapsed);
        black_box(&output);
    }
    let elapsed = started.elapsed();
    Ok(Measurement {
        initialization_ms,
        effective_strength,
        cpu_fallback_allowed,
        statistics: Statistics::calculate(&mut latencies, elapsed),
    })
}

// Generate one repeatable second outside all timings; never benchmark only silence.
pub fn synthetic_input() -> Vec<[f32; FRAME_SIZE]> {
    let mut state = INPUT_SEED;
    (0..SAMPLE_RATE as usize / FRAME_SIZE)
        .map(|frame| {
            std::array::from_fn(|index| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let noise = f64::from(state >> 8) / 16_777_216.0 * 2.0 - 1.0;
                let time = (frame * FRAME_SIZE + index) as f64 / f64::from(SAMPLE_RATE);
                let phase = std::f64::consts::TAU * time;
                (0.18 * (220.0 * phase).sin() + 0.08 * (880.0 * phase).sin() + 0.04 * noise) as f32
            })
        })
        .collect()
}
