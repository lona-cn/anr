#[cfg(any(
    target_os = "windows",
    target_os = "linux",
    target_os = "macos",
    all(target_arch = "wasm32", target_os = "unknown")
))]
mod shared;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod web;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub(crate) use web::Gpu;

#[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
mod native;
#[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
pub(crate) use native::Gpu;

#[cfg(not(any(
    target_os = "windows",
    target_os = "linux",
    target_os = "macos",
    all(target_arch = "wasm32", target_os = "unknown")
)))]
pub(crate) struct Gpu;

#[cfg(not(any(
    target_os = "windows",
    target_os = "linux",
    target_os = "macos",
    all(target_arch = "wasm32", target_os = "unknown")
)))]
impl Gpu {
    pub(crate) fn new(_: anr_dfn3::Model) -> anr_core::Result<Self> {
        Err(anr_dfn3::backend(
            "native wgpu DFN3 requires Windows, Linux, or macOS",
        ))
    }
}

#[cfg(not(any(
    target_os = "windows",
    target_os = "linux",
    target_os = "macos",
    all(target_arch = "wasm32", target_os = "unknown")
)))]
impl anr_dfn3::Executor for Gpu {
    fn frame(
        &mut self,
        _: &[f32; anr_dfn3::FRAME_SIZE],
        _: &mut [f32; anr_dfn3::FRAME_SIZE],
        _: f32,
        _: bool,
    ) -> anr_core::Result<()> {
        Err(anr_dfn3::backend(
            "native wgpu DFN3 is unavailable on this platform",
        ))
    }
}
