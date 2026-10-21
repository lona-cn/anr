/// Prepares ONNX Runtime without configuring the process-wide environment.
///
/// With `load-dynamic`, loads `ORT_DYLIB_PATH` or ORT's platform-default library
/// name when the variable is unset or empty. An already loaded runtime is reused;
/// failed loads can be retried. With static linking, this does nothing.
///
/// This does not create an environment. Applications configure logging,
/// execution providers, and global thread pools through [`ort::init`], then
/// build and retain their own environment for session creation.
///
/// # Errors
/// Returns [`anr_core::Error::Backend`] with the original dynamic-loading error
/// as its source if the runtime library cannot be loaded.
pub fn initialize_runtime() -> anr_core::Result<()> {
    #[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
    {
        #[cfg(target_os = "windows")]
        const DEFAULT_LIBRARY: &str = "onnxruntime.dll";
        #[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
        const DEFAULT_LIBRARY: &str = "libonnxruntime.so";
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        const DEFAULT_LIBRARY: &str = "libonnxruntime.dylib";

        let configured_path = std::env::var("ORT_DYLIB_PATH").ok();
        let path = configured_path
            .as_deref()
            .filter(|path| !path.is_empty())
            .unwrap_or(DEFAULT_LIBRARY);
        // init_from loads the library before returning this builder. Discard it
        // rather than creating an environment owned by the library.
        let _ = ort::init_from(path).map_err(anr_core::Error::backend)?;
    }
    Ok(())
}
