//! Audio noise reduction through a backend-independent interface.
//!
//! No inference backend is enabled by default. The default `download-models` and
//! `copy-models` features prepare DFN3 for [`default_model_path`].
//! Enable `ort` for `DFN3`, or select a compute backend with its matching feature.

pub use anr_core::{Error, NoiseReduction, Result};

mod models;
pub use models::default_model_path;

/// ONNX Runtime backend and its runtime configuration APIs.
#[cfg(feature = "ort")]
pub mod ort {
    pub use anr_ort::ort::*;
    pub use anr_ort::{DFN3, ModelTarget, initialize_runtime, prepare_model, session_from_file};
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

/// wgpu Compute backend: native `DFN3`, or browser `web::DFN3` on WebAssembly.
#[cfg(feature = "wgpu")]
pub mod wgpu {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    pub use anr_wgpu::DFN3;
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    pub use anr_wgpu::web;
}

/// Native Windows D3D12 Compute backend for the supported DFN3 model.
#[cfg(feature = "dx")]
pub mod dx {
    pub use anr_dx::DFN3;
}

/// Native CUDA compute for the supported DFN3 model, without ONNX Runtime.
#[cfg(feature = "cuda")]
pub mod cuda {
    pub use anr_cuda::DFN3;
}

/// Native TensorRT neural inference with CUDA audio DSP, without ONNX Runtime.
#[cfg(feature = "tensorrt")]
pub mod tensorrt {
    pub use anr_tensorrt::DFN3;
}

/// Native OpenVINO inference with explicitly selected devices.
#[cfg(feature = "openvino")]
pub mod openvino {
    pub use anr_openvino::{DFN3, Device};
}

/// Native OpenCL hardware-GPU compute, without ONNX Runtime.
#[cfg(feature = "opencl")]
pub mod opencl {
    pub use anr_opencl::DFN3;
}
