//! Hash-locked DFN3 inference using native Windows D3D12 compute.
//!
//! System D3DCompiler 47 compiles the model's HLSL at initialization; no
//! external shader compiler, DirectML, ONNX Runtime, or CPU fallback is used.
//! One private worker owns the device, resident state, and synchronous readback.

#[cfg(target_os = "windows")]
mod gpu;
#[cfg(not(target_os = "windows"))]
mod gpu {
    use anr_core::Result;
    use anr_dfn3::{Executor, FRAME_SIZE, Model, backend};

    pub(crate) struct Gpu;
    impl Executor for Gpu {
        fn new(_model: Model) -> Result<Self> {
            Err(backend(
                "native D3D12 inference is supported only on Windows",
            ))
        }
        fn frame(
            &mut self,
            _input: &[f32; FRAME_SIZE],
            _output: &mut [f32; FRAME_SIZE],
            _attenuation: f32,
            _reset: bool,
        ) -> Result<()> {
            Err(backend(
                "native D3D12 inference is supported only on Windows",
            ))
        }
    }
}

use anr_core::{NoiseReduction, Result};
use anr_dfn3::Denoiser;
use std::path::Path;

/// One continuous 48 kHz mono DFN3 stream on a hardware D3D12 device.
///
/// Calls are synchronous. A partial final frame is zero-padded, with no delay
/// compensation or end flush. Invalid input leaves output and state unchanged.
/// The denoiser can move between threads; the device remains on its worker.
pub struct DFN3 {
    inner: Denoiser,
}

impl DFN3 {
    pub const SAMPLE_RATE: u32 = Denoiser::SAMPLE_RATE;
    pub const FRAME_SIZE: usize = Denoiser::FRAME_SIZE;
    pub const STATE_SIZE: usize = Denoiser::STATE_SIZE;

    /// Validates the supported model, then initializes hardware D3D12 compute.
    /// Other models and platforms return an error; WARP is never selected.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            inner: Denoiser::from_file::<gpu::Gpu>(path)?,
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
