use crate::args::{Backend, Provider};
use anr::NoiseReduction;
#[cfg(any(feature = "ort", feature = "gl"))]
use anyhow::Context;
use anyhow::{Result, bail};
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug)]
pub struct EngineSpec {
    pub sample_rate: u32,
    pub frame_samples: usize,
}

pub struct EnginePool {
    model: PathBuf,
    strength: f32,
    spec: EngineSpec,
    reducers: Vec<Box<dyn NoiseReduction>>,
    resources: BackendResources,
}

enum BackendResources {
    #[cfg(feature = "ort")]
    Ort {
        environment: anr::ort::environment::Environment,
        provider: Provider,
    },
    #[cfg(feature = "gl")]
    Gl,
}

impl BackendResources {
    fn new(backend: Backend, provider: Option<Provider>) -> Result<(Self, EngineSpec)> {
        match backend {
            Backend::Ort => {
                let provider = provider.unwrap_or(Provider::Cpu);
                if !provider.compiled() {
                    bail!(
                        "provider {} 未编译或不支持当前平台；请查看 anr backends，并启用对应 ort feature",
                        provider.name()
                    );
                }
                #[cfg(feature = "ort")]
                {
                    anr::ort::initialize_runtime().context("无法初始化 ONNX Runtime")?;
                    let environment = anr::ort::init()
                        .build()
                        .context("无法创建 ONNX Runtime 环境")?;
                    Ok((
                        Self::Ort {
                            environment,
                            provider,
                        },
                        EngineSpec {
                            sample_rate: anr::ort::DFN3::SAMPLE_RATE,
                            frame_samples: anr::ort::DFN3::FRAME_SIZE,
                        },
                    ))
                }
                #[cfg(not(feature = "ort"))]
                bail!("此程序未启用 ORT 后端；请以 ort feature 重新构建")
            }
            Backend::Gl => {
                if provider.is_some() {
                    bail!("--provider 只适用于 ORT，不能与 --backend gl 同用");
                }
                #[cfg(feature = "gl")]
                {
                    Ok((
                        Self::Gl,
                        EngineSpec {
                            sample_rate: anr::gl::DFN3::SAMPLE_RATE,
                            frame_samples: anr::gl::DFN3::FRAME_SIZE,
                        },
                    ))
                }
                #[cfg(not(feature = "gl"))]
                bail!("此程序未启用 GL 后端；请以 gl feature 重新构建")
            }
        }
    }

    fn load(&self, model: &Path, strength: f32) -> Result<Box<dyn NoiseReduction>> {
        #[cfg(any(feature = "ort", feature = "gl"))]
        match self {
            #[cfg(feature = "ort")]
            Self::Ort {
                environment,
                provider,
            } => load_ort_model(environment, model, *provider, strength),
            #[cfg(feature = "gl")]
            Self::Gl => {
                let mut reducer = anr::gl::DFN3::from_file(model)
                    .with_context(|| format!("无法以 GL 后端加载模型 {}", model.display()))?;
                reducer.set_factor(strength);
                Ok(Box::new(reducer))
            }
        }
        #[cfg(not(any(feature = "ort", feature = "gl")))]
        {
            let _ = (model, strength);
            match *self {}
        }
    }
}

impl EnginePool {
    pub fn new(
        model: &Path,
        backend: Backend,
        provider: Option<Provider>,
        strength: f32,
    ) -> Result<Self> {
        let (resources, spec) = BackendResources::new(backend, provider)?;
        let reducer = resources.load(model, strength)?;
        Ok(Self {
            model: model.to_owned(),
            resources,
            strength,
            spec,
            reducers: vec![reducer],
        })
    }

    pub fn prepare(&mut self, channels: usize) -> Result<()> {
        if channels == 0 {
            bail!("输入没有音频声道");
        }
        while self.reducers.len() < channels {
            let reducer = self.resources.load(&self.model, self.strength)?;
            self.reducers.push(reducer);
        }
        self.reducers.truncate(channels);
        // pipeline::process resets once for each new stream, not each packet.
        Ok(())
    }

    pub fn spec(&self) -> EngineSpec {
        self.spec
    }
    pub fn reducers(&mut self) -> &mut [Box<dyn NoiseReduction>] {
        &mut self.reducers
    }
}

#[cfg(feature = "ort")]
fn load_ort_model(
    environment: &anr::ort::environment::Environment,
    model: &Path,
    provider: Provider,
    strength: f32,
) -> Result<Box<dyn NoiseReduction>> {
    let mut reducer = if provider == Provider::Cpu {
        anr::ort::DFN3::from_file(environment, model)
            .with_context(|| format!("无法加载模型 {}", model.display()))?
    } else {
        let session = (|| -> anr::ort::Result<anr::ort::session::Session> {
            let mut builder = anr::ort::session::Session::builder(environment)?
                .with_no_environment_execution_providers()?
                .with_independent_thread_pool()?
                .with_intra_threads(1)?
                .with_intra_op_spinning(false)?
                .with_inter_op_spinning(false)?
                .with_parallel_execution(false)?;
            if provider == Provider::Directml {
                builder = builder.with_memory_pattern(false)?;
            }
            // Explicit GPU requests must not silently execute the graph on CPU.
            // XNNPACK is itself a CPU provider, with CPU kernels for other ops.
            if provider != Provider::Xnnpack {
                builder = builder.with_config_entry("session.disable_cpu_ep_fallback", "1")?;
            }
            builder
                .with_execution_providers([dispatch(provider)
                    .map_err(|e| anr::ort::Error::new(e.to_string()))?
                    .error_on_failure()])?
                .commit_from_file(model)
        })()
        .with_context(|| {
            format!(
                "无法以 provider {} 初始化模型 {}",
                provider.name(),
                model.display()
            )
        })?;
        anr::ort::DFN3::new(session).context("模型的 DFN3 输入输出接口不兼容")?
    };
    reducer.set_factor(strength);
    Ok(Box::new(reducer))
}

#[cfg(feature = "ort")]
fn dispatch(provider: Provider) -> Result<anr::ort::ep::ExecutionProviderDispatch> {
    use anr::ort::ep;
    Ok(match provider {
        Provider::Cpu => ep::CPU::default().build(),
        #[cfg(feature = "ort-cuda")]
        Provider::Cuda => ep::CUDA::default().build(),
        #[cfg(feature = "ort-tensorrt")]
        Provider::Tensorrt => ep::TensorRT::default().build(),
        #[cfg(all(feature = "ort-directml", target_os = "windows"))]
        Provider::Directml => ep::DirectML::default().build(),
        #[cfg(all(feature = "ort-coreml", target_os = "macos"))]
        Provider::Coreml => ep::CoreML::default().build(),
        #[cfg(feature = "ort-xnnpack")]
        Provider::Xnnpack => ep::XNNPACK::default().build(),
        _ => bail!("provider {} 未编译或不支持当前平台", provider.name()),
    })
}

#[derive(Serialize)]
pub struct BackendInfo {
    pub name: &'static str,
    pub compiled: bool,
    pub runtime_available: Option<bool>,
    pub providers: Vec<ProviderInfo>,
}

#[derive(Serialize)]
pub struct ProviderInfo {
    pub name: &'static str,
    pub compiled: bool,
    pub runtime_available: Option<bool>,
}

pub fn backends() -> Vec<BackendInfo> {
    vec![
        BackendInfo {
            name: "ort",
            compiled: cfg!(feature = "ort"),
            runtime_available: None,
            providers: Provider::ALL
                .iter()
                .map(|provider| ProviderInfo {
                    name: provider.name(),
                    compiled: provider.compiled(),
                    runtime_available: None,
                })
                .collect(),
        },
        BackendInfo {
            name: "gl",
            compiled: cfg!(feature = "gl"),
            runtime_available: None,
            providers: Vec::new(),
        },
    ]
}
