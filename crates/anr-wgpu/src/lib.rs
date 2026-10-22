//! Hash-locked DeepFilterNet3 inference with native wgpu compute shaders.
//!
//! The external ONNX file supplies fixed parameters, not a runtime graph. Native
//! hardware compute executes the complete model, FFT, and audio DSP; no software
//! adapter or CPU inference fallback is used. A private worker owns every GPU
//! resource for its entire lifetime, while [`DFN3`] can move between threads.
mod gpu;

use anr_core::{NoiseReduction, Result};
use anr_dfn3::{Denoiser, Model};
use std::path::Path;

/// One continuous 48 kHz mono DFN3 stream on a native hardware adapter.
///
/// Calls are synchronous and return only after GPU readback. Short final frames
/// are zero-padded and advance a full model frame. Invalid input leaves state
/// and output unchanged. No model-delay compensation or end flush is added.
pub struct DFN3 {
    inner: Denoiser,
}

impl DFN3 {
    /// Required mono sample rate.
    pub const SAMPLE_RATE: u32 = Denoiser::SAMPLE_RATE;
    /// Samples consumed by one model frame.
    pub const FRAME_SIZE: usize = Denoiser::FRAME_SIZE;
    /// Recurrent audio, feature, and network state values.
    pub const STATE_SIZE: usize = Denoiser::STATE_SIZE;

    /// Validates the supported model before creating the worker-owned device.
    ///
    /// Supported native platforms are Windows, Linux, and macOS. A hardware
    /// Vulkan, Direct3D 12, or Metal adapter is required; no surface is created.
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
