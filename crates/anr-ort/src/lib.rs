//! ONNX Runtime backend for audio noise reduction.
//!
//! [`DFN3`] implements [`anr_core::NoiseReduction`]. The runtime is re-exported
//! as [`ort`] so consumers can configure a [`ort::session::Session`] without
//! accidentally selecting an incompatible ORT crate version.

mod dfn3;
mod runtime;

pub use dfn3::DFN3;
pub use ort;
pub use runtime::initialize_runtime;
