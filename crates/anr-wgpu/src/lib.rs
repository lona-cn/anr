//! Hash-locked DeepFilterNet3 inference with wgpu compute shaders.
//!
//! The external ONNX file supplies fixed parameters, not a runtime graph.
//! Hardware compute executes the complete model, FFT, and audio DSP; no software
//! adapter or CPU inference fallback is used. Native streams use a private GPU
//! worker and a synchronous API; browser streams use the asynchronous WebGPU API.
mod gpu;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod native;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use native::DFN3;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod wasm;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub mod web;
