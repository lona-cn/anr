//! ANR-Lite-48: a differentiable full-band speech enhancement model.
//!
//! This crate defines the architecture and its streaming signal processing.
//! The optional `train` feature adds a VoiceBank-DEMAND training binary.
//! No pretrained weights or automatic dataset downloads are included.
//!
//! [`AnrLite48`] is a regular Burn [`burn::module::Module`]: use `train()` for
//! autodiff, `valid()` for inference, and its record methods for checkpoints.
//! [`AudioProcessor::forward`] returns an aligned waveform; explicit streaming
//! uses [`AudioProcessor::forward_frame`] and [`AudioProcessor::flush`].

mod audio;
mod fft;
mod layers;
mod network;
mod spectral;

/// Model sample rate in hertz.
pub const SAMPLE_RATE: u32 = 48_000;
/// Audio samples consumed by one streaming step.
pub const FRAME_SIZE: usize = 480;
/// Length of the analysis and synthesis transform.
pub const FFT_SIZE: usize = 960;
/// Non-redundant real FFT bins.
pub const FREQ_BINS: usize = 481;
/// Perceptual full-band feature and gain count.
pub const ERB_BANDS: usize = 64;
/// Low-frequency bins receiving complex residual filtering.
pub const DF_BINS: usize = 96;
/// Complex temporal filter taps, current spectrum first.
pub const DF_TAPS: usize = 5;
/// Future spectral frames relative to the enhanced target.
pub const LOOKAHEAD_FRAMES: usize = 2;
/// Streaming sample delay, excluding collection of the current input hop.
pub const AUDIO_DELAY_SAMPLES: usize = (LOOKAHEAD_FRAMES + 1) * FRAME_SIZE;

pub use burn;

pub use audio::{AudioProcessor, AudioState};
pub use fft::Fft960;
pub use network::{AnrLite48, ModelOutput, ModelState};
pub use spectral::{SpectralProcessor, SpectralState};
