//! Audio noise reduction through a backend-independent interface.
//!
//! The default build exports only [`NoiseReduction`], [`Error`], and [`Result`].
//! Enable the `ort` feature for `DFN3` and the `ort` backend module.

pub use anr_core::{Error, NoiseReduction, Result};

/// ONNX Runtime backend and its runtime configuration APIs.
#[cfg(feature = "ort")]
pub mod ort {
    pub use anr_ort::DFN3;
    pub use anr_ort::ort::*;
}

#[cfg(feature = "ort")]
pub use anr_ort::DFN3;
