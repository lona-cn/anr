//! Fixed DeepFilterNet3 inference using native OpenCL 1.2 GPU compute.
//!
//! OpenCL is loaded dynamically; no build-time SDK or display is required.
//! A hardware GPU and its OpenCL C compiler must be installed. CPU devices,
//! ONNX Runtime and implicit backend fallbacks are never used. All recurrent
//! state and inference/DSP computation remain on the private worker's GPU.
mod gpu;

use anr_core::{NoiseReduction, Result};
use anr_dfn3::{Denoiser, Model};
use std::path::Path;

/// One continuous 48 kHz mono stream using a hardware OpenCL GPU.
///
/// `run` waits for completion and downloads only the output audio frame. A
/// partial frame is zero-padded and advances a whole 480-sample frame, without
/// latency compensation or an implicit final flush. Validation errors leave
/// output and stream state untouched. Recreate the reducer after device errors.
/// Reset preserves the selected strength and clears both GPU state buffers.
pub struct DFN3 {
    inner: Denoiser,
}

impl DFN3 {
    /// Required mono sample rate in hertz.
    pub const SAMPLE_RATE: u32 = Denoiser::SAMPLE_RATE;
    /// Samples consumed per complete frame.
    pub const FRAME_SIZE: usize = Denoiser::FRAME_SIZE;
    /// Packed recurrent-state values in the supported export.
    pub const STATE_SIZE: usize = Denoiser::STATE_SIZE;

    /// Validates the supported model before loading the OpenCL ICD/device.
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
