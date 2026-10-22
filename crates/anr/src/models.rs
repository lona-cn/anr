use crate::{Error, Result};
use std::path::{Path, PathBuf};

/// Finds the DFN3 model prepared by `download-models` or `copy-models`.
///
/// Checks `models/dfn3.onnx` beside the running executable first, then the
/// absolute model path recorded at build time. This does not depend on the
/// working directory and never downloads at runtime. The executable-relative
/// path also works with bundled models when both features are disabled.
///
/// `ANR_MODEL_PATH` can override the source file at build time. Applications
/// supplying their own model can continue passing a path directly to a backend.
///
/// # Errors
///
/// Returns an error if the executable path cannot be determined or neither
/// model file exists. When distributing an executable to another machine,
/// include its `models/dfn3.onnx`; the build cache is not portable.
pub fn default_model_path() -> Result<PathBuf> {
    let executable = std::env::current_exe().map_err(Error::backend)?;
    if let Some(directory) = executable.parent() {
        let model = directory.join("models/dfn3.onnx");
        if model.is_file() {
            return Ok(model);
        }
    }
    if let Some(model) = option_env!("ANR_DEFAULT_MODEL_PATH") {
        let model = Path::new(model);
        if model.is_file() {
            return Ok(model.to_path_buf());
        }
    }
    Err(Error::InvalidModel(
        "DFN3 model not found: place models/dfn3.onnx beside the executable, enable download-models, or supply a model path explicitly".into(),
    ))
}
