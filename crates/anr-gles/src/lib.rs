//! Fixed-model DFN3 inference using native OpenGL ES 3.1 compute shaders.
//! Reuses the GL-family executor and shared model without desktop GL or CPU fallback.

use anr_core::{NoiseReduction, Result};
use anr_gl::GlApi;
use std::path::Path;

/// One 48 kHz mono stream with a worker-owned native OpenGL ES context.
/// `run` is synchronous. Short final frames are padded, and model delay is not
/// compensated. The supported external ONNX file supplies parameters only.
pub struct DFN3 {
    inner: anr_gl::DFN3,
}

impl DFN3 {
    /// Required sample rate.
    pub const SAMPLE_RATE: u32 = anr_gl::DFN3::SAMPLE_RATE;
    /// Samples consumed per model step.
    pub const FRAME_SIZE: usize = anr_gl::DFN3::FRAME_SIZE;
    /// Packed recurrent state values.
    pub const STATE_SIZE: usize = anr_gl::DFN3::STATE_SIZE;

    /// Initializes real GLES 3.1 on Windows/WGL or hardware Linux/EGL.
    /// Unsupported models/drivers/platforms fail without another graphics API,
    /// software renderer or CPU inference fallback.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            inner: anr_gl::DFN3::from_file_with_api(path, GlApi::Gles)?,
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
