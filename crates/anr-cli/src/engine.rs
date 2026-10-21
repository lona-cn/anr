use crate::args::Provider;
use anr::NoiseReduction;
#[cfg(feature = "ort")]
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
    provider: Provider,
    strength: f32,
    spec: EngineSpec,
    reducers: Vec<Box<dyn NoiseReduction>>,
    #[cfg(feature = "ort")]
    environment: anr::ort::environment::Environment,
}

impl EnginePool {
    pub fn new(model: &Path, provider: Provider, strength: f32) -> Result<Self> {
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
            let reducer = load_model(&environment, model, provider, strength)?;
            Ok(Self {
                model: model.to_owned(),
                provider,
                strength,
                spec: EngineSpec {
                    sample_rate: anr::DFN3::SAMPLE_RATE,
                    frame_samples: anr::DFN3::FRAME_SIZE,
                },
                reducers: vec![reducer],
                environment,
            })
        }
        #[cfg(not(feature = "ort"))]
        {
            let _ = (model, strength);
            bail!("此程序未启用 ORT 后端；请以 ort feature 重新构建")
        }
    }

    pub fn prepare(&mut self, channels: usize) -> Result<()> {
        if channels == 0 {
            bail!("输入没有音频声道");
        }
        while self.reducers.len() < channels {
            #[cfg(feature = "ort")]
            let reducer = load_model(&self.environment, &self.model, self.provider, self.strength)?;
            #[cfg(not(feature = "ort"))]
            let reducer = load_model(&self.model, self.provider, self.strength)?;
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
fn load_model(
    environment: &anr::ort::environment::Environment,
    model: &Path,
    provider: Provider,
    strength: f32,
) -> Result<Box<dyn NoiseReduction>> {
    let mut reducer = if provider == Provider::Cpu {
        anr::DFN3::from_file(environment, model)
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
        anr::DFN3::new(session).context("模型的 DFN3 输入输出接口不兼容")?
    };
    reducer.set_factor(strength);
    Ok(Box::new(reducer))
}

#[cfg(not(feature = "ort"))]
fn load_model(
    _model: &Path,
    _provider: Provider,
    _strength: f32,
) -> Result<Box<dyn NoiseReduction>> {
    bail!("此程序未启用 ORT 后端")
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
    pub providers: Vec<ProviderInfo>,
}

#[derive(Serialize)]
pub struct ProviderInfo {
    pub name: &'static str,
    pub compiled: bool,
    pub runtime_available: Option<bool>,
}

pub fn backends() -> Vec<BackendInfo> {
    vec![BackendInfo {
        name: "ort",
        compiled: cfg!(feature = "ort"),
        providers: Provider::ALL
            .iter()
            .map(|provider| ProviderInfo {
                name: provider.name(),
                compiled: provider.compiled(),
                runtime_available: None,
            })
            .collect(),
    }]
}
