use crate::args::Backend;
use anr_core::{Error, Result};
use anr_ort::{DFN3, ort};
use std::path::Path;

pub fn load(
    backend: Backend,
    model: &Path,
    environment: &mut Option<ort::environment::Environment>,
) -> Result<DFN3> {
    let provider = dispatch(backend)?;
    if environment.is_none() {
        anr_ort::initialize_runtime()?;
        *environment = Some(
            ort::init()
                .with_name("anr-bench")
                .build()
                .map_err(Error::backend)?,
        );
    }
    // Retain the one process-wide environment even when session creation fails.
    let environment = environment.as_ref().expect("initialized ORT environment");
    let session = (|| -> ort::Result<ort::session::Session> {
        let mut builder = ort::session::Session::builder(environment)?
            .with_no_environment_execution_providers()?
            .with_independent_thread_pool()?
            .with_intra_threads(1)?
            .with_intra_op_spinning(false)?
            .with_inter_op_spinning(false)?
            .with_parallel_execution(false)?;
        if backend == Backend::OrtDirectml {
            builder = builder.with_memory_pattern(false)?;
        }
        // Allow ORT's per-operator CPU placement, but never ignore EP registration errors.
        builder
            .with_execution_providers([provider.error_on_failure()])?
            .commit_from_file(model)
    })()
    .map_err(Error::backend)?;
    DFN3::new(session)
}

fn dispatch(backend: Backend) -> Result<ort::ep::ExecutionProviderDispatch> {
    match backend {
        #[cfg(feature = "ort-cuda")]
        Backend::OrtCuda => Ok(ort::ep::CUDA::default().build()),
        #[cfg(feature = "ort-tensorrt")]
        Backend::OrtTensorrt => Ok(ort::ep::TensorRT::default().build()),
        #[cfg(all(feature = "ort-directml", target_os = "windows"))]
        Backend::OrtDirectml => Ok(ort::ep::DirectML::default().build()),
        #[cfg(not(target_os = "windows"))]
        Backend::OrtDirectml => Err(anr_dfn3::backend("ort-directml 仅支持 Windows")),
        _ => Err(anr_dfn3::backend(format!(
            "后端 {} 未编译；请启用同名 Cargo feature",
            backend.name()
        ))),
    }
}
