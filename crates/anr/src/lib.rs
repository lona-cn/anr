//! Audio noise reduction through a backend-independent interface.
//!
//! No inference backend is enabled by default. The default `download-models` and
//! `copy-models` features prepare DFN3 for [`default_model_path`].
//! Enable `ort` for `DFN3`, or `gl`, `gles`, `vk`, `wgpu`, or `dx` for a native backend.

pub use anr_core::{Error, NoiseReduction, Result};

mod models;
pub use models::default_model_path;

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
    pub use anr_gl::{DFN3, GlApi};
}

/// Native OpenGL ES Compute backend for the supported DFN3 model.
#[cfg(feature = "gles")]
pub mod gles {
    pub use anr_gles::DFN3;
}

/// Native Vulkan Compute backend for the supported DFN3 model.
#[cfg(feature = "vk")]
pub mod vk {
    pub use anr_vk::DFN3;
}

/// Native wgpu Compute backend for the supported DFN3 model.
#[cfg(feature = "wgpu")]
pub mod wgpu {
    pub use anr_wgpu::DFN3;
}

/// Native Windows D3D12 Compute backend for the supported DFN3 model.
#[cfg(feature = "dx")]
pub mod dx {
    pub use anr_dx::DFN3;
}
