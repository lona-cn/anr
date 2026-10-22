//! Fixed DeepFilterNet3 inference using native Vulkan 1.1 compute on Windows/Linux.
//!
//! The supported external ONNX model is a hash-locked parameter container, not
//! an executable CPU/ONNX Runtime graph. All inference, DSP and recurrent state
//! stay on a hardware GPU owned by a private worker; no display is required.
#[cfg(any(target_os = "windows", target_os = "linux"))]
mod gpu;

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
mod gpu {
    use anr_core::Result;
    use anr_dfn3::{Executor, FRAME_SIZE, Model, backend};

    pub(crate) struct Gpu;
    impl Gpu {
        pub(crate) fn new(_: Model) -> Result<Self> {
            Err(backend(
                "native Vulkan DFN3 is supported only on Windows and Linux",
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
                "native Vulkan DFN3 is supported only on Windows and Linux",
            ))
        }
    }
}

use anr_core::{NoiseReduction, Result};
use anr_dfn3::{Denoiser, Model};
use std::path::Path;

/// One continuous 48 kHz mono stream with hardware Vulkan compute inference.
///
/// `run` waits for GPU completion and readback. A partial 480-sample frame is
/// zero-padded and advances a whole frame. No latency compensation/end flush
/// is added. Validation failures leave output/state untouched. Reset before
/// reuse after an execution error; device loss requires recreating the stream.
/// The Vulkan loader and a hardware Vulkan 1.1 compute device are required.
/// CPU devices and implicit backend fallbacks are never used.
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

    /// Validates the supported model before opening a hardware Vulkan device.
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
