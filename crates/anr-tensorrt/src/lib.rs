//! DFN3 inference with two FP32 TensorRT standard-layer neural engines and
//! native CUDA FFT/DSP, captured together in parity-specific CUDA graphs.
//!
//! Build with TensorRT 10 and CUDA SDKs (`TENSORRT_ROOT`, `CUDA_ROOT`, or
//! `CUDA_PATH`). The private native shim is loaded only when constructing this
//! backend, so missing runtimes cannot prevent other backends from starting.
//! `ANR_TENSORRT_SHIM` selects the shim; otherwise the executable directory and
//! build output are searched. On Windows `ANR_TENSORRT_RUNTIME` selects the TRT
//! DLL directory and `CUDA_ROOT` selects CUDA DLLs; PATH is also supported.
//! Linux uses the platform loader paths. Deploy the shim beside the executable
//! along with the required licensed SDK runtime libraries. A CUDA hardware GPU
//! supported by TensorRT is required; there is no CPU/ORT fallback.
mod ffi;

use anr_core::{NoiseReduction, Result};
use anr_dfn3::{Denoiser, Model};
use std::path::Path;

/// One continuous 48 kHz mono stream. Initialization builds two strongly typed
/// FP32 engines (TF32 disabled), allocates all device memory, warms execution,
/// captures the complete frame, and restores zero business state. `run` waits
/// for GPU completion. Partial frames pad with zero and advance a whole frame.
/// Validation failures preserve output and state; a fatal execution error
/// requires creating a new reducer. Reset preserves attenuation.
pub struct DFN3 {
    inner: Denoiser,
}

impl DFN3 {
    /// Required mono sample rate in hertz.
    pub const SAMPLE_RATE: u32 = Denoiser::SAMPLE_RATE;
    /// Samples in a complete frame.
    pub const FRAME_SIZE: usize = Denoiser::FRAME_SIZE;
    /// Logical recurrent state values.
    pub const STATE_SIZE: usize = Denoiser::STATE_SIZE;

    /// Validates the hash-locked model before creating CUDA/TensorRT resources.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let model = Model::load_aligned(path.as_ref(), 64)?;
        Ok(Self {
            inner: Denoiser::spawn(move || ffi::Gpu::new(model))?,
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
