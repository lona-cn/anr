//! Audio noise reduction through a backend-independent interface.
//!
//! The default build exports only [`NoiseReduction`], [`Error`], and [`Result`].
//! Enable the `ort` feature for `DFN3` and the `ort` backend module.
//! Enable `gl` for `gl::DFN3`, the fixed-model OpenGL Compute backend.

pub use anr_core::{Error, NoiseReduction, Result};

/// ONNX Runtime backend and its runtime configuration APIs.
#[cfg(feature = "ort")]
pub mod ort {
    pub use anr_ort::ort::*;
    pub use anr_ort::{DFN3, initialize_runtime};
}

#[cfg(feature = "ort")]
pub use anr_ort::DFN3;

/// OpenGL Compute backend for the supported DFN3 model.
#[cfg(feature = "gl")]
pub mod gl {
    pub use anr_gl::DFN3;
}
