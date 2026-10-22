# anr

Rust audio denoising with DeepFilterNet3, native compute backends, and a file/stream CLI.

[English](#english) · [简体中文](#简体中文) · [Benchmarks](https://github.com/lona-cn/anr/blob/v0.2.0/benchmark.md) · [Changelog](https://github.com/lona-cn/anr/blob/v0.2.0/CHANGELOG.md)

## English

`anr` provides a stateful denoising interface for **48 kHz mono `f32` audio**. Run the supported DeepFilterNet3 (DFN3) model through ONNX Runtime or a model-specific native backend. The [`anr` CLI](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-cli/README.md) adds audio decoding, resampling, independent per-channel processing, batch jobs, and raw PCM streams. It does not provide microphone capture, virtual audio devices, or drivers.

[ANR-Lite-48](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-lite/README.md) is a separate Burn-based model and training pipeline. It includes differentiable audio processing, CPU/wgpu execution, VoiceBank-DEMAND training, checkpoint rendering, and objective evaluation tools. It ships no pretrained weights and is not a backend of the DFN3 CLI.

The registry examples below apply after **0.2.0** is published to crates.io. Until then, use the local-checkout alternatives. Repository documentation links target the matching `v0.2.0` tag and become available when that tag is published.

### Install and denoise a file

**Application ZIP:** check [GitHub Releases](https://github.com/lona-cn/anr/releases) for an application archive for your platform. The release workflow packages the executable, CPU ONNX Runtime, `models/dfn3.onnx`, and license notices. Extract the entire archive, then run from its directory:

```sh
./anr denoise voice.wav --output clean.wav
```

On Windows, use `.\anr.exe`. Keep the bundled runtime and model with the executable. The separate model asset is not an application archive; packaging and system requirements are documented in the [CLI guide](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-cli/README.md).

**crates.io (after the 0.2.0 release):** use the latest stable Rust toolchain and the platform's native build prerequisites:

```text
cargo install anr-cli --version 0.2.0 --locked
anr denoise voice.wav --output clean.wav
anr denoise a.flac b.mp3 --output-dir cleaned
anr inspect voice.wav --json
anr backends --json
```

**Local checkout:** from the repository root, replace the installation command with:

```text
cargo install --path crates/anr-cli --locked
```

The default CLI build uses CPU ORT. It prepares the model at build time, so `--model` is optional; `--model PATH` overrides automatic discovery. The first build needs network access unless the model and runtime are already available locally. Intel macOS needs a supplied ORT runtime rather than the default prebuilt distribution; see the [CLI guide](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-cli/README.md) for that configuration and offline builds.

By default, file processing preserves sample rate, channel count, and frame count. Output is WAV or raw f32le. The CLI guide covers formats, stdout pipelines, output protection, cancellation, provider selection, and latency behavior.

### Rust quickstart

In an application outside this workspace, add the registry dependency and explicitly enable ORT:

```toml
[dependencies]
anr = { version = "0.2.0", features = ["ort"] }
```

For an unpublished local checkout, use its actual path instead:

```toml
anr = { path = "/path/to/anr/crates/anr", features = ["ort"] }
```

```rust,no_run
use anr::{DFN3, NoiseReduction, Result};

fn main() -> Result<()> {
    anr::ort::initialize_runtime()?;
    let environment = anr::ort::init().build().map_err(anr::Error::backend)?;
    let mut denoiser = DFN3::from_file(&environment, anr::default_model_path()?)?;
    denoiser.set_factor(80.0);

    // Replace silence with one frame of 48 kHz mono audio.
    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    denoiser.run(DFN3::SAMPLE_RATE, &input, &mut output)?;

    // Start another recording without changing the strength setting.
    denoiser.reset();
    Ok(())
}
```

Keep one denoiser per continuous stream. DFN3 processes **480 samples (10 ms)** per model frame; use multiples of 480 for continuous input. A short final frame is zero-padded and advances a full frame of state. `run` requires nonempty input, 48,000 Hz, and equal input/output lengths. The native synchronous interface is `Send` and object-safe, but waits for processing to finish; it is not a nonblocking audio-callback API.

`set_factor(0..=100)` controls the attenuation limit, not a dry/wet mix. Zero is not bypass. DFN3 library processing does not compensate model latency or provide an end-of-stream flush; the CLI also does not infer an alignment offset. These limitations matter when integrating live audio or comparing output with a clean reference. See the [library guide](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr/README.md) for state, errors, strength rounding, and the raw-audio example.

### Backends and defaults

| Component | Default behavior |
| --- | --- |
| `anr` library | No execution backend. Enable `ort` or a native backend feature explicitly. |
| `anr-cli` | ORT enabled, CPU provider selected. A native-only build still needs `--backend NAME`. |
| Model preparation | `download-models` and `copy-models` are enabled by default in the facade and CLI. Download, SHA-256 verification, caching, and copying happen at build time, not runtime. |
| ORT runtime option | The facade's default `ort-download-binaries` does not itself enable `ort`; the CLI's option does. `ort-load-dynamic` supports a supplied runtime. |

`anr::default_model_path()` checks `models/dfn3.onnx` beside the executable, then the source path recorded during the build. Model bytes are not embedded in Rust packages. For deployment, include the model rather than relying on a build-machine cache. `ANR_MODEL_PATH` selects a local build-time source; feature combinations, cache paths, and offline setup are in the [library guide](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr/README.md).

| Feature / implementation | Execution route |
| --- | --- |
| `ort` / `anr-ort` | ONNX Runtime; CPU by default, configurable sessions and execution providers. |
| `gl`, `gles` | Native OpenGL 4.3 or OpenGL ES 3.1+ compute on Windows/Linux; GLES requires a real ES context. |
| `vk` | Native Vulkan 1.1 compute on Windows/Linux, independent of wgpu. |
| `wgpu` | Native Vulkan/DX12/Metal; asynchronous browser WebGPU on `wasm32-unknown-unknown`. |
| `dx` | Native Windows Direct3D 12 compute, not DirectML. |
| `cuda` | CUDA Driver API and NVRTC, with GPU-resident state and replay graphs. |
| `tensorrt` | Native TensorRT engines plus CUDA DSP/FFT; TensorRT/CUDA C++ SDKs required. |
| `openvino` | Native OpenVINO ONNX import; explicit CPU/GPU/NPU selection, CPU by default; C++ SDK required. |
| `opencl` | OpenCL 1.2 hardware GPU execution. |

The nine native backends are independent of ORT and accept only the pinned DFN3 model identified in the [model notice](https://github.com/lona-cn/anr/blob/v0.2.0/.github/models/NOTICE.md). OpenVINO imports its graph; the other native implementations execute a fixed model-specific schedule. The ORT adapter instead validates the DFN3 tensor contract; it is not a general-purpose ONNX audio interface.

`ort-cuda`, `ort-tensorrt`, `ort-directml`, `ort-coreml`, and `ort-xnnpack` enable **ORT execution-provider support**, not the similarly named native backends. Provider features do not install drivers or select a GPU. ORT may place individual operators on CPU; failure to register the requested provider remains an error. Native GPU initialization does not silently fall back to CPU. OpenVINO's explicit CPU selection is a device choice, not fallback; GPU/NPU support depends on the device and graph, and NPU execution may use lower precision.

For a native Vulkan application on Windows/Linux that supplies its own model file:

```toml
anr = { version = "0.2.0", default-features = false, features = ["vk"] }
```

Browser WebGPU uses `anr::wgpu::web::DFN3` and asynchronous model-byte loading/processing rather than the synchronous `NoiseReduction` trait. It needs HTTPS or localhost, a hardware WebGPU adapter, and separately served model bytes. The [library guide](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr/README.md) covers package generation and the [module Worker example](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-wgpu/examples/web/worker.js); it does not supply microphone or AudioWorklet buffering.

For backend platform restrictions, SDK/runtime deployment, constructors, and model ports, use the [library guide](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr/README.md). For timing methodology, historical measurements, and reproduction commands, use [benchmark.md](https://github.com/lona-cn/anr/blob/v0.2.0/benchmark.md); a 10 ms model frame is not a processing-time or end-to-end latency guarantee.

### Crates and documentation

| Crate / document | Purpose |
| --- | --- |
| [`anr`](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr/README.md) | Application-facing facade, feature selection, model preparation, native/backend setup, browser integration. |
| [`anr-core`](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-core/README.md) | Runtime-independent `NoiseReduction`, `Error`, and `Result`. |
| [`anr-ort`](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-ort/README.md) | DFN3 tensor contract, runtime initialization, custom sessions/providers, model preparation. |
| [`anr-cli`](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-cli/README.md) · [architecture](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-cli/DESIGN.md) | File, batch, inspection, and raw-stream commands; audio pipeline and output lifecycle. |
| [`anr-gl`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-gl), [`anr-gles`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-gles), [`anr-vk`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-vk), [`anr-wgpu`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-wgpu), [`anr-dx`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-dx) | Graphics-compute implementations; setup is documented in the facade guide. |
| [`anr-cuda`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-cuda), [`anr-tensorrt`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-tensorrt), [`anr-openvino`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-openvino), [`anr-opencl`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-opencl) | Native compute/runtime implementations; setup is documented in the facade guide. |
| [`anr-dfn3`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-dfn3) | Shared fixed-model parameters, execution plan, worker, and shader sources; not a device backend. |
| [`anr-lite`](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-lite/README.md) | ANR-Lite-48 architecture, streaming/batch DSP, training, checkpoints, and DFN3 quality comparison. |
| [`anr-bench`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-bench) · [benchmark guide](https://github.com/lona-cn/anr/blob/v0.2.0/benchmark.md) | Workspace-only streaming benchmark; not included in application ZIPs. |
| [Dataset catalog](https://github.com/lona-cn/anr/blob/v0.2.0/datasets/DATASETS.md) | Dataset formats, access, and license conditions. |
| [Release guide](https://github.com/lona-cn/anr/blob/v0.2.0/RELEASING.md) · [CI workflow](https://github.com/lona-cn/anr/blob/v0.2.0/.github/workflows/ci.yml) | Packaging, version/tag policy, verification, and GitHub publication. |

### Development

Use the latest Rust stable; this workspace does not maintain a fixed MSRV. Start with the crate guide for the component you are changing. The [CI workflow](https://github.com/lona-cn/anr/blob/v0.2.0/.github/workflows/ci.yml) defines SDK-free workspace checks, the WASM build, and native application packaging for Windows x64, Linux x64, and macOS arm64/x64. Ignored GPU tests require suitable hardware and an explicit local run; TensorRT and OpenVINO need SDK-enabled builds. The [release guide](https://github.com/lona-cn/anr/blob/v0.2.0/RELEASING.md) describes the full distribution procedure; the workflow does not publish crates.io packages.

Model/backend regression fixtures and hardware requirements are documented in the library guide; ANR-Lite development and evaluation have their own [guide](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-lite/README.md). See [CHANGELOG.md](https://github.com/lona-cn/anr/blob/v0.2.0/CHANGELOG.md) for version history.

### License

Code is **MIT OR Apache-2.0**: choose either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE). The exact pinned model has its own [notice and authorization](https://github.com/lona-cn/anr/blob/v0.2.0/.github/models/NOTICE.md), identified by the [model manifest](https://github.com/lona-cn/anr/blob/v0.2.0/.github/models/dfn3.json); that authorization does not cover arbitrary exports or weights.

Decoders, ONNX protobuf definitions, ORT, native runtimes, and execution-provider dependencies have their own terms. Consult [third-party notices](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-cli/THIRD-PARTY-NOTICES.md) when distributing binaries; the project's dual license does not replace those obligations.

## 简体中文

`anr` 提供面向 **48 kHz、单声道 `f32` 音频**的有状态降噪接口，可通过 ONNX Runtime 或模型专用原生后端运行受支持的 DeepFilterNet3（DFN3）模型。[`anr` 命令行工具](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-cli/README.md) 负责音频解码、重采样、各声道独立处理、批处理和 raw PCM 流，不提供麦克风采集、虚拟声卡或驱动。

[ANR-Lite-48](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-lite/README.md) 是独立的 Burn 模型与训练流程，包含可微音频处理、CPU/wgpu 执行、VoiceBank-DEMAND 训练、checkpoint 音频导出和客观指标评估工具。不附带预训练权重，也不是 DFN3 CLI 的可选后端。

下方 registry 示例适用于 **0.2.0 发布到 crates.io 后**；发布前可使用本地源码示例。仓库文档链接指向对应的 `v0.2.0` tag，该 tag 发布后才能访问。

### 安装并处理文件

**应用 ZIP：** 在 [GitHub Releases](https://github.com/lona-cn/anr/releases) 中查看适合当前平台的应用包。发版工作流将可执行文件、CPU ONNX Runtime、`models/dfn3.onnx` 和许可说明打包在一起。完整解压后，进入解压目录运行：

```sh
./anr denoise voice.wav --output clean.wav
```

Windows 使用 `.\anr.exe`。运行库和模型应随可执行文件一起保留；独立模型资产不是应用安装包。包结构及系统要求见 [CLI 文档](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-cli/README.md)。

**crates.io（0.2.0 发布后）：** 准备最新 Rust stable 和目标平台的原生构建依赖：

```text
cargo install anr-cli --version 0.2.0 --locked
anr denoise voice.wav --output clean.wav
anr denoise a.flac b.mp3 --output-dir cleaned
anr inspect voice.wav --json
anr backends --json
```

**本地源码：** 在仓库根目录使用以下安装命令：

```text
cargo install --path crates/anr-cli --locked
```

CLI 默认构建使用 CPU ORT，在构建时准备模型，因此 `--model` 可省略；`--model PATH` 可覆盖自动查找结果。首次构建需要网络，除非模型与运行库已在本地准备好。Intel macOS 需要自备 ORT 运行库，不能直接使用默认预编译分发；配置与离线构建方法见 [CLI 文档](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-cli/README.md)。

文件处理默认保留输入采样率、声道数和帧数，输出 WAV 或 raw f32le。格式、stdout 管道、输出保护、取消、provider 选择和延迟语义均在 CLI 文档中说明。

### Rust 快速开始

在工作区以外的应用中，添加 registry 依赖并显式启用 ORT：

```toml
[dependencies]
anr = { version = "0.2.0", features = ["ort"] }
```

使用尚未发布的本地源码时，改为实际源码路径：

```toml
anr = { path = "/path/to/anr/crates/anr", features = ["ort"] }
```

```rust,no_run
use anr::{DFN3, NoiseReduction, Result};

fn main() -> Result<()> {
    anr::ort::initialize_runtime()?;
    let environment = anr::ort::init().build().map_err(anr::Error::backend)?;
    let mut denoiser = DFN3::from_file(&environment, anr::default_model_path()?)?;
    denoiser.set_factor(80.0);

    // 用一帧实际的 48 kHz 单声道音频替换静音。
    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    denoiser.run(DFN3::SAMPLE_RATE, &input, &mut output)?;

    // 开始处理另一段录音，保留强度设置。
    denoiser.reset();
    Ok(())
}
```

每路连续音频使用独立实例。DFN3 每个模型帧处理 **480 个采样（10 ms）**，连续输入宜按 480 的整数倍提交；末尾短帧会补零，并推进完整一帧的状态。`run` 要求输入非空、采样率为 48,000 Hz，且输入输出长度相同。原生同步接口支持 `Send` 和 trait object，但调用会等待处理完成，不是非阻塞音频 callback。

`set_factor(0..=100)` 控制衰减上限，不是 dry/wet 混合比例，零也不表示旁路。DFN3 库不补偿模型延迟，不提供结束 flush；CLI 也不会推测对齐偏移。实时音频接入和与 clean 参考比较时都需要考虑这些限制。状态、错误、强度取整和 raw 音频示例见 [库文档](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr/README.md)。

### 后端与默认行为

| 组件 | 默认行为 |
| --- | --- |
| `anr` 库 | 不启用执行后端，必须显式选择 `ort` 或原生后端 feature。 |
| `anr-cli` | 启用 ORT，选择 CPU provider；只编入原生后端时仍须传 `--backend NAME`。 |
| 模型准备 | facade 和 CLI 默认启用 `download-models`、`copy-models`，在构建时下载、校验 SHA-256、缓存和复制，不在运行时下载。 |
| ORT 运行库选项 | facade 默认的 `ort-download-binaries` 本身不启用 `ort`，CLI 的同名选项会启用；`ort-load-dynamic` 用于自备运行库。 |

`anr::default_model_path()` 先查找可执行文件旁的 `models/dfn3.onnx`，再使用构建时记录的源路径。模型内容不嵌入 Rust 包；分发程序时应附带模型，不能依赖构建机器的缓存。`ANR_MODEL_PATH` 可指定本地构建时模型来源，feature 组合、缓存目录及离线用法见 [库文档](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr/README.md)。

| Feature / 实现 | 执行方式 |
| --- | --- |
| `ort` / `anr-ort` | ONNX Runtime，默认 CPU，可自定义 Session 和 execution provider。 |
| `gl`、`gles` | Windows/Linux 的原生 OpenGL 4.3 或 OpenGL ES 3.1+ Compute；GLES 要求真实 ES context。 |
| `vk` | Windows/Linux 的原生 Vulkan 1.1 Compute，不经过 wgpu。 |
| `wgpu` | 原生 Vulkan/DX12/Metal；`wasm32-unknown-unknown` 上的异步浏览器 WebGPU。 |
| `dx` | 原生 Windows Direct3D 12 Compute，不是 DirectML。 |
| `cuda` | CUDA Driver API、NVRTC、GPU 常驻状态与 replay graph。 |
| `tensorrt` | 原生 TensorRT engine 与 CUDA DSP/FFT，构建需要 TensorRT/CUDA C++ SDK。 |
| `openvino` | 原生 OpenVINO 导入 ONNX；显式选择 CPU/GPU/NPU，默认 CPU，构建需要 C++ SDK。 |
| `opencl` | OpenCL 1.2 硬件 GPU 执行。 |

九种原生后端均独立于 ORT，只接受 [模型说明](https://github.com/lona-cn/anr/blob/v0.2.0/.github/models/NOTICE.md) 标识的固定 DFN3 模型。OpenVINO 导入其计算图，其余原生实现使用模型专用的固定调度。ORT 适配器则检查 DFN3 的张量接口，也不是任意 ONNX 音频模型的通用运行器。

`ort-cuda`、`ort-tensorrt`、`ort-directml`、`ort-coreml`、`ort-xnnpack` 启用的是 **ORT execution provider 支持**，不是同名原生后端。provider feature 不安装驱动，也不自动选 GPU。ORT 可以将部分算子分配到 CPU，但请求的 provider 注册失败仍报错。原生 GPU 初始化失败不会静默回退 CPU。OpenVINO 显式选择 CPU 属于设备选择而非 fallback；GPU/NPU 可用性取决于设备与计算图，NPU 可能使用较低精度。

Windows/Linux 原生 Vulkan 应用自行提供模型文件时，可使用：

```toml
anr = { version = "0.2.0", default-features = false, features = ["vk"] }
```

浏览器 WebGPU 使用 `anr::wgpu::web::DFN3`，通过异步接口加载模型字节并处理音频，不实现同步 `NoiseReduction` trait。需要 HTTPS 或 localhost、硬件 WebGPU adapter，以及单独提供的模型字节。[库文档](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr/README.md) 介绍包构建方法和 [module Worker 示例](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-wgpu/examples/web/worker.js)，不提供麦克风或 AudioWorklet 缓冲。

后端平台限制、SDK/运行库部署、构造器和模型端口见 [库文档](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr/README.md)。测量方法、历史数据和复现命令见 [benchmark.md](https://github.com/lona-cn/anr/blob/v0.2.0/benchmark.md)；10 ms 的模型帧长不代表处理耗时或端到端延迟保证。

### Crate 与文档导航

| Crate / 文档 | 用途 |
| --- | --- |
| [`anr`](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr/README.md) | 应用入口、feature 选择、模型准备、原生后端配置和浏览器集成。 |
| [`anr-core`](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-core/README.md) | 不依赖推理运行库的 `NoiseReduction`、`Error`、`Result`。 |
| [`anr-ort`](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-ort/README.md) | DFN3 张量约定、运行库初始化、自定义 Session/provider 与模型准备。 |
| [`anr-cli`](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-cli/README.md) · [架构](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-cli/DESIGN.md) | 文件、批处理、信息查询及 raw 流命令；音频管线与输出生命周期。 |
| [`anr-gl`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-gl)、[`anr-gles`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-gles)、[`anr-vk`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-vk)、[`anr-wgpu`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-wgpu)、[`anr-dx`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-dx) | 图形 Compute 实现，配置说明集中在 facade 文档。 |
| [`anr-cuda`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-cuda)、[`anr-tensorrt`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-tensorrt)、[`anr-openvino`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-openvino)、[`anr-opencl`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-opencl) | 原生 Compute/runtime 实现，配置说明集中在 facade 文档。 |
| [`anr-dfn3`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-dfn3) | 共用固定模型参数、执行顺序、worker 与 shader 源码，本身不是设备后端。 |
| [`anr-lite`](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-lite/README.md) | ANR-Lite-48 架构、流式/批量 DSP、训练、checkpoint 和 DFN3 音质比较。 |
| [`anr-bench`](https://github.com/lona-cn/anr/tree/v0.2.0/crates/anr-bench) · [性能文档](https://github.com/lona-cn/anr/blob/v0.2.0/benchmark.md) | 工作区内的连续帧 benchmark，不包含在应用 ZIP 中。 |
| [数据集目录](https://github.com/lona-cn/anr/blob/v0.2.0/datasets/DATASETS.md) | 数据格式、获取方式及许可条件。 |
| [发版指南](https://github.com/lona-cn/anr/blob/v0.2.0/RELEASING.md) · [CI 工作流](https://github.com/lona-cn/anr/blob/v0.2.0/.github/workflows/ci.yml) | 打包、版本/tag 策略、验证和 GitHub 发布。 |

### 开发

使用最新 Rust stable；工作区不维护固定 MSRV。从所修改组件的文档开始。[CI 工作流](https://github.com/lona-cn/anr/blob/v0.2.0/.github/workflows/ci.yml) 定义不需要厂商 SDK 的工作区检查、WASM 构建，以及 Windows x64、Linux x64、macOS arm64/x64 的原生应用打包。Ignored GPU 测试需要在适合的硬件上显式运行；TensorRT、OpenVINO 需要配置 SDK 构建。[发版指南](https://github.com/lona-cn/anr/blob/v0.2.0/RELEASING.md) 介绍完整分发流程；工作流不发布 crates.io 包。

模型/后端回归数据及硬件要求见库文档，ANR-Lite 开发与评估见其 [专门文档](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-lite/README.md)。版本历史见 [CHANGELOG.md](https://github.com/lona-cn/anr/blob/v0.2.0/CHANGELOG.md)。

### 许可

代码采用 **MIT OR Apache-2.0**，可任选 [MIT](LICENSE-MIT) 或 [Apache-2.0](LICENSE-APACHE)。固定模型的 [说明与授权](https://github.com/lona-cn/anr/blob/v0.2.0/.github/models/NOTICE.md) 只适用于 [模型清单](https://github.com/lona-cn/anr/blob/v0.2.0/.github/models/dfn3.json) 标识的文件，不覆盖其他导出文件或权重。

解码器、ONNX protobuf 定义、ORT、原生运行库和 execution provider 依赖各有许可要求。分发二进制时请查阅 [第三方说明](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-cli/THIRD-PARTY-NOTICES.md)，项目双许可不替代这些义务。
