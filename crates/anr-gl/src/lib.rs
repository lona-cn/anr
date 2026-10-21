//! Fixed-model DFN3 inference using native desktop OpenGL 4.3 or OpenGL ES 3.1.
//! The model, shaders and stream semantics are shared with the other native
//! compute backends; GL resource ownership and synchronization remain local.
mod context;
mod gpu;

use anr_core::{NoiseReduction, Result};
use anr_dfn3::Denoiser;
pub(crate) use anr_dfn3::{FRAME_SIZE, STATE_SIZE, backend};
use std::path::Path;

/// Native graphics API to request. Context creation never falls back to another API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlApi {
    /// Desktop OpenGL 4.3 Core or newer.
    Desktop,
    /// Native OpenGL ES 3.1 or newer, not desktop ES shader compatibility.
    Gles,
}

/// One 48 kHz mono stream with a worker-owned native OpenGL context.
/// `run` is synchronous. Short final frames are padded, and model delay is not
/// compensated. The supported external ONNX file supplies parameters only.
pub struct DFN3 {
    inner: Denoiser,
}
impl DFN3 {
    /// Required sample rate.
    pub const SAMPLE_RATE: u32 = Denoiser::SAMPLE_RATE;
    /// Samples consumed per model step.
    pub const FRAME_SIZE: usize = Denoiser::FRAME_SIZE;
    /// Packed recurrent state values.
    pub const STATE_SIZE: usize = Denoiser::STATE_SIZE;
    /// Initializes desktop GL 4.3 on Windows/WGL or hardware Linux/EGL.
    /// Unsupported models/drivers/platforms fail without a CPU fallback.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_file_with_api(path, GlApi::Desktop)
    }
    /// Initializes the requested native API on Windows/WGL or hardware Linux/EGL.
    /// GLES requires a real ES 3.1 context; neither API nor software fallbacks are used.
    pub fn from_file_with_api(path: impl AsRef<Path>, api: GlApi) -> Result<Self> {
        let inner = match api {
            GlApi::Desktop => Denoiser::from_file::<gpu::Gpu<false>>(path)?,
            GlApi::Gles => Denoiser::from_file::<gpu::Gpu<true>>(path)?,
        };
        Ok(Self { inner })
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
