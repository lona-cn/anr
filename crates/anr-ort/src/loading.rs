use crate::{ModelTarget, prepare_model};
use anr_core::{Error, Result};
use ort::{
    AsPointer, ep::ExecutionProviderDispatch, ortsys, session::Session,
    session::builder::SessionBuilder,
};
use std::path::Path;

/// Loads a model with the requested providers and their required graph preparation.
///
/// CUDA and TensorRT models are prepared in memory; the source file is never changed.
/// Other providers retain ONNX Runtime's file-loading behavior. Provider registration
/// errors are returned, while ordinary per-operator CPU placement remains allowed.
pub fn session_from_file(
    mut builder: SessionBuilder,
    path: impl AsRef<Path>,
    providers: &[ExecutionProviderDispatch],
) -> Result<Session> {
    let path = path.as_ref();
    let target = if providers
        .iter()
        .any(|provider| provider.name() == "TensorrtExecutionProvider")
    {
        Some(ModelTarget::TensorRT)
    } else if providers
        .iter()
        .any(|provider| provider.name() == "CUDAExecutionProvider")
    {
        Some(ModelTarget::Cuda)
    } else {
        None
    };
    let bytes = target
        .map(|_| std::fs::read(path).map_err(Error::backend))
        .transpose()?;
    let prepared = bytes
        .as_deref()
        .zip(target)
        .map(|(bytes, target)| prepare_model(bytes, target))
        .transpose()?;
    if prepared.is_some() && !has_external_directory(&builder).map_err(Error::backend)? {
        // CreateSessionFromArray has no source filename. Supply its directory
        // only when the caller has not explicitly configured external weights.
        let absolute_path = std::path::absolute(path).map_err(Error::backend)?;
        let directory = absolute_path
            .parent()
            .and_then(Path::to_str)
            .ok_or_else(|| {
                Error::InvalidModel(
                    "CUDA/TensorRT in-memory loading requires a UTF-8 model directory to resolve external weights".into(),
                )
            })?;
        builder = builder
            .with_config_entry(
                "session.model_external_initializers_file_folder_path",
                directory,
            )
            .map_err(|error| Error::backend(ort::Error::<()>::from(error)))?;
    }
    for provider in providers {
        builder = builder
            .with_execution_providers([provider.clone().error_on_failure()])
            .map_err(|error| Error::backend(ort::Error::<()>::from(error)))?;
    }
    match prepared {
        Some(model) => builder
            .commit_from_memory(model.as_ref())
            .map_err(Error::backend),
        None => builder.commit_from_file(path).map_err(Error::backend),
    }
}

fn has_external_directory(builder: &SessionBuilder) -> ort::Result<bool> {
    let mut configured = 0;
    ortsys![unsafe HasSessionConfigEntry(
        builder.ptr(),
        c"session.model_external_initializers_file_folder_path".as_ptr(),
        &mut configured
    )?];
    Ok(configured != 0)
}
