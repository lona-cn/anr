#[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
mod native;
#[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
pub(crate) use native::Gpu;

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
pub(crate) struct Gpu;

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
impl anr_dfn3::Executor for Gpu {
    fn new(_: anr_dfn3::Model) -> anr_core::Result<Self> {
        Err(anr_dfn3::backend(
            "native wgpu DFN3 requires Windows, Linux, or macOS",
        ))
    }

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
