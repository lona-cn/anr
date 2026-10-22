//! Fixed DeepFilterNet3 inference using native NVIDIA CUDA and replay graphs.
//!
//! Model weights, intermediate tensors and recurrent states remain on the GPU.
//! CUDA Driver and NVRTC are loaded dynamically; no build-time CUDA SDK is
//! required. Put the CUDA NVRTC library on PATH (Windows) or LD_LIBRARY_PATH
//! (Linux). Driver API 12.0+ and NVRTC 12/13 are required.
//! The installed driver must support the selected NVRTC's generated PTX, and
//! NVRTC must support the GPU architecture. No CPU fallback is used.
#[cfg(any(target_os = "windows", target_os = "linux"))]
mod gpu;
#[cfg(any(target_os = "windows", target_os = "linux"))]
mod runtime;

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
mod gpu {
    use anr_core::Result;
    use anr_dfn3::{Executor, FRAME_SIZE, Model, backend};
    pub(crate) struct Gpu;
    impl Gpu {
        pub(crate) fn new(_: Model) -> Result<Self> {
            Err(backend(
                "native CUDA DFN3 is supported only on Windows and Linux",
            ))
        }
    }
    impl Executor for Gpu {
        fn frame(
            &mut self,
            _: &[f32; FRAME_SIZE],
            _: &mut [f32; FRAME_SIZE],
            _: f32,
            _: bool,
        ) -> Result<()> {
            Err(backend(
                "native CUDA DFN3 is supported only on Windows and Linux",
            ))
        }
    }
}

use anr_core::{NoiseReduction, Result};
use anr_dfn3::{Denoiser, Model};
use std::path::Path;

/// One continuous 48 kHz mono stream on an NVIDIA GPU.
///
/// `run` synchronously waits for graph completion and copies only the audio
/// frame back to the CPU. Partial frames are zero-padded and advance one full
/// frame. Reset preserves strength. Validation errors preserve output/state.
/// Any CUDA execution error permanently invalidates the stream; recreate it
/// rather than attempting to reuse possibly corrupted recurrent state.
pub struct DFN3 {
    inner: Denoiser,
}

impl DFN3 {
    /// Required mono sample rate in hertz.
    pub const SAMPLE_RATE: u32 = Denoiser::SAMPLE_RATE;
    /// Samples consumed by one complete model frame.
    pub const FRAME_SIZE: usize = Denoiser::FRAME_SIZE;
    /// Recurrent state values in the supported model export.
    pub const STATE_SIZE: usize = Denoiser::STATE_SIZE;

    /// Validates the supported model before loading CUDA or opening a device.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let model = Model::load(path.as_ref())?;
        Ok(Self {
            inner: Denoiser::spawn(move || gpu::Gpu::new(model))?,
        })
    }
}

impl NoiseReduction for DFN3 {
    fn run(&mut self, sample_rate: u32, input: &[f32], output: &mut [f32]) -> Result<()> {
        self.inner.run(sample_rate, input, output)
    }
    fn factor(&self) -> f32 {
        self.inner.factor()
    }
    fn set_factor(&mut self, value: f32) {
        self.inner.set_factor(value);
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
}
