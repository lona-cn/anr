//! Browser WebGPU inference for the fixed DeepFilterNet3 model.
//!
//! This API is asynchronous and independent of the native worker-backed stream.
//! It never creates threads or blocks the browser while waiting for the GPU.
use crate::gpu::Gpu;
use anr_core::{Error, Result};
use anr_dfn3::{FRAME_SIZE, Model};

/// One continuous 48 kHz mono DFN3 stream on a browser WebGPU device.
///
/// A mutable borrow serializes operations on the stream. Short final frames are
/// zero-padded and advance a full model frame. No delay compensation or end
/// flush is added. Dropping an in-flight run releases its mapping and causes the
/// next run to reset recurrent state; dropping the stream never waits for GPU work.
pub struct DFN3 {
    gpu: Gpu,
    input: [f32; FRAME_SIZE],
    output: [f32; FRAME_SIZE],
    attenuation: f32,
    reset_pending: bool,
}

impl DFN3 {
    /// Required mono sample rate.
    pub const SAMPLE_RATE: u32 = anr_dfn3::SAMPLE_RATE;
    /// Samples consumed by one model frame.
    pub const FRAME_SIZE: usize = FRAME_SIZE;
    /// Recurrent audio, feature, and network state values.
    pub const STATE_SIZE: usize = anr_dfn3::STATE_SIZE;

    /// Validates the hash-locked ONNX bytes, uploads weights, and waits for
    /// WebGPU resource and shader validation. No software fallback is used.
    pub async fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let model = Model::from_bytes(bytes)?;
        Ok(Self {
            gpu: Gpu::new(model).await?,
            input: [0.0; FRAME_SIZE],
            output: [0.0; FRAME_SIZE],
            attenuation: 3.0,
            reset_pending: false,
        })
    }

    /// Processes mono samples, asynchronously awaiting GPU readback per frame.
    ///
    /// Invalid input leaves output and stream state unchanged. An execution
    /// error may follow already completed frames; the failing frame is not
    /// published, and subsequent execution starts from reset state.
    pub async fn run(&mut self, sample_rate: u32, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.is_empty() {
            return Err(Error::EmptyInput);
        }
        if sample_rate != Self::SAMPLE_RATE {
            return Err(Error::InvalidSampleRate {
                expected: Self::SAMPLE_RATE,
                actual: sample_rate,
            });
        }
        if input.len() != output.len() {
            return Err(Error::OutputLengthMismatch {
                input: input.len(),
                output: output.len(),
            });
        }
        for (input, output) in input.chunks(FRAME_SIZE).zip(output.chunks_mut(FRAME_SIZE)) {
            self.input[..input.len()].copy_from_slice(input);
            self.input[input.len()..].fill(0.0);
            let reset = self.reset_pending;
            // Set before the first possible GPU submission/await. This remains
            // true on both errors and cancellation; only a complete frame clears it.
            self.reset_pending = true;
            self.gpu
                .frame(&self.input, &mut self.output, self.attenuation, reset)
                .await?;
            output.copy_from_slice(&self.output[..output.len()]);
            self.reset_pending = false;
        }
        Ok(())
    }

    /// Current strength, rounded to the same percentage scale as native DFN3.
    pub fn factor(&self) -> f32 {
        (self.attenuation / 60.0 * 100.0).round()
    }

    /// Clamps to 0..=100 and rounds attenuation to whole decibels; ignores NaN.
    pub fn set_factor(&mut self, value: f32) {
        if !value.is_nan() {
            self.attenuation = (value.clamp(0.0, 100.0) / 100.0 * 60.0).round();
        }
    }

    /// Resets audio, feature, and network state before the next frame, retaining
    /// the configured strength. This does not submit work or block the browser.
    pub fn reset(&mut self) {
        self.reset_pending = true;
    }
}
