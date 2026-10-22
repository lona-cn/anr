//! Fixed DFN3 model data, dispatch schedule and synchronous stream handling.
//!
//! This crate contains no device API or inference runtime. The external ONNX
//! file is a hash-locked parameter container. Backends execute the same fixed
//! compute shaders, while owning their native devices and synchronization.
mod model;
mod network;
pub mod shaders;
mod weights;
mod worker;

pub use model::Model;
pub use weights::model_bytes;
pub use worker::{Denoiser, Executor};

/// Required mono sample rate.
pub const SAMPLE_RATE: u32 = 48_000;
/// Samples consumed per model frame.
pub const FRAME_SIZE: usize = 480;
/// Packed recurrent audio, feature and neural-network state values.
pub const STATE_SIZE: usize = 45_304;
/// Size of one packed uniform parameter block, in bytes.
pub const PARAMETER_BYTES: u32 = 128;

/// The three fixed compute programs used by every backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderKind {
    Network,
    Dsp,
    Fft,
}
impl ShaderKind {
    /// Stable pipeline order for native backends.
    pub const ALL: [Self; 3] = [Self::Network, Self::Dsp, Self::Fft];
    /// Index in a three-element pipeline array.
    pub const fn index(self) -> usize {
        match self {
            Self::Network => 0,
            Self::Dsp => 1,
            Self::Fft => 2,
        }
    }
}

/// One dispatch in the statically constructed model schedule.
#[derive(Clone, Debug)]
pub struct Dispatch {
    /// Compute program.
    pub shader: ShaderKind,
    /// Exactly 128 bytes, interpreted as eight packed uint4 values.
    pub params: [u32; 32],
    /// Number of workgroups in X; Y and Z are always one.
    pub groups: u32,
}

/// Wraps an owned backend diagnostic using the common error type.
pub fn backend(message: impl Into<String>) -> anr_core::Error {
    anr_core::Error::backend(std::io::Error::other(message.into()))
}
