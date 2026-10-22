# anr-bench

[English](#english) | [简体中文](#简体中文)

## English

A standalone CLI for comparing synchronous, streaming DeepFilterNet3 inference. It generates repeatable **48 kHz mono `f32`** input, processes **480 samples per frame**, and tests selected backends serially. It does not process your audio files, measure denoising quality, capture audio devices, or download models.

### Install and run

Use the latest stable Rust toolchain and your platform's native linker/build tools:

```text
cargo install anr-bench --version 0.2.0 --locked
anr-bench --help
```

`cargo install` builds an optimized executable by default. Keep Cargo's installation directory on `PATH`. From a checkout, run `cargo install --path crates/anr-bench --locked` at the repository root instead. Avoid `--debug` for performance measurements; the report records whether debug assertions were enabled.

Prepare the [pinned DFN3 model](https://github.com/lona-cn/anr/releases/download/model-dfn3-fe5eb64fa2e4/dfn3.onnx) and check its SHA-256:

```text
fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9
```

The model is not in the source package or installed executable. Its [provenance and redistribution terms](https://github.com/lona-cn/anr/blob/v0.2.0/.github/models/NOTICE.md) are separate from the source-code license. Use an absolute model path to run from any directory, for example with hardware Vulkan on Windows/Linux:

```text
anr-bench --model /absolute/path/to/dfn3.onnx --backend vk --warmup 100 --frames 1000 --strength 80 --json
```

Replace the path with your actual file; on Windows, a quoted path such as `"C:/models/dfn3.onnx"` works. Relative paths resolve from the current working directory, not the executable directory. The default is `models/dfn3.onnx`; no automatic model discovery or download is performed.

### Backends, features and runtimes

**`gl`, `vk`, `dx`, `cuda` and `opencl` are always compiled in**, not optional Cargo features. The default feature list is empty and `--no-default-features` does not remove these backends. Defaults at runtime are **`--backend gl,vk,dx`**; select only backends your machine supports. In particular, `dx` fails on non-Windows platforms, so use an explicit list on Linux. The default native backends target Windows/Linux and require real GPU implementations, not CPU/software adapters.

| `--backend` value | Extra Cargo feature | Build/runtime requirements |
| --- | --- | --- |
| `gl` | None | Desktop OpenGL 4.3; Windows WGL or Linux hardware EGL context. |
| `vk` | None | Hardware Vulkan 1.1 and its loader on Windows/Linux. |
| `dx` | None | Windows D3D12 compute and system D3DCompiler 47; no WARP. |
| `cuda` | None | NVIDIA Driver API 12.0+ and NVRTC 12/13, with a driver that supports the compiler's PTX; no build-time CUDA SDK. |
| `opencl` | None | OpenCL 1.2 hardware GPU ICD and compiler; a loader alone is insufficient. |
| `tensorrt` | `tensorrt` | C++17, TensorRT 10 and CUDA SDKs; compatible NVIDIA GPU and vendor runtimes. |
| `openvino` | `openvino` | C++17 and OpenVINO C++ SDK; Runtime, ONNX frontend and requested device plugin. CPU is the explicit default device. |
| `ort-cuda` | `ort-cuda` plus an ORT runtime option | Compatible ONNX Runtime CUDA provider, CUDA/cuDNN libraries and NVIDIA GPU. |
| `ort-tensorrt` | `ort-tensorrt` plus an ORT runtime option | Compatible ONNX Runtime TensorRT provider and its vendor dependencies. |
| `ort-directml` | `ort-directml` plus an ORT runtime option | Windows and a compatible ONNX Runtime DirectML provider. |

Native CUDA/TensorRT are independent of the similarly named ORT providers. This tool has no GLES, wgpu, ORT CPU, CoreML or XNNPACK measurement entry. Native GPU failures do not fall back to CPU; OpenVINO requests the device explicitly and does not switch devices on failure. ORT may place individual operators on CPU, but provider registration failures remain errors.

For an application-supplied ORT shared library, install the provider feature with `ort-load-dynamic`, for example:

```text
cargo install anr-bench --version 0.2.0 --locked --features ort-cuda,ort-load-dynamic
```

Before running, set `ORT_DYLIB_PATH` to the actual API-27-compatible shared-library **file**, then select `--backend ort-cuda`. Alternatively, `ort-download-binaries` enables the upstream runtime downloader; it does not install drivers or guarantee that the downloaded distribution contains every requested GPU provider. `ort` alone does not enable a provider. Keep the core runtime and provider libraries from the same distribution.

TensorRT needs `TENSORRT_ROOT` and `CUDA_ROOT` or `CUDA_PATH`; OpenVINO needs `OPENVINO_ROOT` or `OpenVINO_DIR`. Their builds generate private shared shims. When installing these features, use a persistent `--target-dir` if relying on recorded build paths, or deploy the shim beside the executable before removing build outputs. Vendor runtime dependencies and OpenVINO frontend/device plugins must also be discoverable. `cargo install` does not bundle a relocatable SDK/runtime distribution. Follow the [backend deployment guide](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr/README.md) and [ORT runtime guide](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-ort/README.md); do not enable `--all-features` without preparing all SDKs and runtimes.

### Options and results

| Option | Meaning / default |
| --- | --- |
| `--model PATH` | Model file; default `models/dfn3.onnx`. |
| `--backend LIST` | Comma-separated or repeated names, measured in the supplied order; default `gl,vk,dx`. |
| `--warmup N` | Extra warmup frames per backend, outside measurement; default `100`, may be `0`. |
| `--frames N` | Measured frames per backend; default `1000`, must be positive. |
| `--strength VALUE` | Finite `0..=100`; default `80`. Actual backend strength is reported separately. |
| `--openvino-device DEVICE` | `CPU`, `GPU`, `GPU.n` or `NPU`; default `CPU`, only used by OpenVINO. |
| `--json` | JSON on stdout instead of the table; progress and diagnostics stay on stderr. |

One backend error does not stop later entries. The final exit status is **1 if any backend fails**; an error row has no fabricated timing values. Thus a nonzero exit status can accompany a useful JSON report containing both successful and failed entries. Argument parsing errors exit before measurement.

Initialization is timed separately from warmup and measured calls. Warmup state and input position carry into measurement without a reset. Per-frame latency covers the entire synchronous `run`, including host/device transfers and completion waits, not just GPU kernels. P50/P95/P99 use nearest rank. FPS and RTF use the full measured loop's wall time; `RTF < 1` alone is not a real-time deadline guarantee. Neither this tool nor those timings measure device I/O, resampling, application buffering, model algorithmic latency or perceptual quality.

JSON records configuration, input seed, OS/architecture, debug-assertion status, per-backend initialization/strength/statistics, and OpenVINO's actual execution device. `cpu_fallback_allowed` describes policy, not the measured fraction of CPU work; it is false for native OpenVINO even when CPU was explicitly selected. Record model hash, hardware, drivers and runtime/SDK versions separately: the report does not discover these. For formulas, historical measurements and full checkout reproduction commands, see the [benchmark report](https://github.com/lona-cn/anr/blob/v0.2.0/benchmark.md). Package-local implementation: [measurement](src/benchmark.rs), [statistics](src/statistics.rs), [report](src/report.rs).

### License and tests

Project code is **MIT OR Apache-2.0**, at your option: [LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE). Models, drivers, SDKs and Rust dependencies retain their own terms; see the [third-party distribution notes](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-cli/THIRD-PARTY-NOTICES.md) before redistributing binaries.

The source package includes the statistics/CLI tests and a tiny self-authored ONNX control graph, not the full DFN3 model. The ignored three-provider ORT regression requires Windows, its Cargo features and matching GPU/runtime libraries; it verifies provider handling, not DFN3 quality or the historical performance figures. Ordinary tests do not require GPU execution.

## 简体中文

独立的 DeepFilterNet3 连续帧推理 benchmark CLI。工具生成固定种子的 **48 kHz 单声道 `f32`** 输入，以 **480 采样/帧**处理，按指定顺序串行测量后端。不处理用户音频文件，不评价降噪音质，不采集音频设备，也不下载模型。

### 安装与运行

准备最新 Rust stable 和目标平台的原生链接/构建工具：

```text
cargo install anr-bench --version 0.2.0 --locked
anr-bench --help
```

`cargo install` 默认生成优化后的可执行文件，将 Cargo 安装目录加入 `PATH`。使用源码 checkout 时，在仓库根目录改用 `cargo install --path crates/anr-bench --locked`。性能测量不要加 `--debug`；报告会记录构建是否启用 debug assertions。

另行准备[固定 DFN3 模型](https://github.com/lona-cn/anr/releases/download/model-dfn3-fe5eb64fa2e4/dfn3.onnx)，核对 SHA-256：

```text
fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9
```

源码包和安装后的程序均不含模型。模型的[来源与再分发条件](https://github.com/lona-cn/anr/blob/v0.2.0/.github/models/NOTICE.md)独立于源码许可。提供绝对模型路径后，可以在任意目录运行。例如，在支持硬件 Vulkan 的 Windows/Linux 上：

```text
anr-bench --model /absolute/path/to/dfn3.onnx --backend vk --warmup 100 --frames 1000 --strength 80 --json
```

替换为实际模型路径；Windows 可使用 `"C:/models/dfn3.onnx"` 这样的带引号路径。相对路径以当前工作目录为准，不以可执行文件目录为准。默认路径为 `models/dfn3.onnx`，工具不会自动查找或下载模型。

### 后端、features 与运行库

**`gl`、`vk`、`dx`、`cuda`、`opencl` 始终编入程序**，不是可关闭的 Cargo features。默认 feature 列表为空，`--no-default-features` 不会移除这五个后端。运行时默认选择 **`--backend gl,vk,dx`**；应显式选择本机支持的后端。`dx` 只支持 Windows，Linux 不应直接使用默认测量列表。默认原生后端面向 Windows/Linux，需要真实硬件 GPU，不使用 CPU 或软件 adapter。

| `--backend` 名称 | 额外 Cargo feature | 构建/运行要求 |
| --- | --- | --- |
| `gl` | 无 | OpenGL 4.3；Windows WGL 或 Linux 硬件 EGL context。 |
| `vk` | 无 | Windows/Linux 的硬件 Vulkan 1.1 及 loader。 |
| `dx` | 无 | Windows D3D12 Compute 与系统 D3DCompiler 47，不使用 WARP。 |
| `cuda` | 无 | NVIDIA Driver API 12.0+、NVRTC 12/13；驱动须支持实际生成的 PTX，构建时不要求 CUDA SDK。 |
| `opencl` | 无 | OpenCL 1.2 硬件 GPU ICD 和编译器，仅安装 loader 不够。 |
| `tensorrt` | `tensorrt` | C++17、TensorRT 10 与 CUDA SDK；相容的 NVIDIA GPU 和厂商运行库。 |
| `openvino` | `openvino` | C++17、OpenVINO C++ SDK；Runtime、ONNX frontend 及所选设备 plugin。明确默认选择 CPU。 |
| `ort-cuda` | `ort-cuda` 加 ORT 运行库选项 | 相容的 ONNX Runtime CUDA provider、CUDA/cuDNN 和 NVIDIA GPU。 |
| `ort-tensorrt` | `ort-tensorrt` 加 ORT 运行库选项 | 相容的 ONNX Runtime TensorRT provider 及其厂商依赖。 |
| `ort-directml` | `ort-directml` 加 ORT 运行库选项 | Windows 及相容的 ONNX Runtime DirectML provider。 |

原生 CUDA/TensorRT 独立于同名 ORT provider。本工具没有 GLES、wgpu、ORT CPU、CoreML、XNNPACK 的测量入口。原生 GPU 后端失败不改用 CPU；OpenVINO 明确指定设备，失败不切换设备。ORT 可以按算子使用 CPU，但 provider 注册失败仍报错。

自备 ORT 共享库时，将 provider feature 与 `ort-load-dynamic` 一起启用，例如：

```text
cargo install anr-bench --version 0.2.0 --locked --features ort-cuda,ort-load-dynamic
```

运行前将 `ORT_DYLIB_PATH` 指向实际兼容 API 27 的共享库**文件**，再选择 `--backend ort-cuda`。也可用 `ort-download-binaries` 启用上游运行库下载器，但它不安装驱动，也不保证下载分发包含所有 GPU provider。单独启用 `ort` 不会启用 provider。core runtime 与 provider 库应来自同一分发版本。

TensorRT 构建需要 `TENSORRT_ROOT` 和 `CUDA_ROOT` / `CUDA_PATH`；OpenVINO 需要 `OPENVINO_ROOT` 或 `OpenVINO_DIR`。两者会生成私有共享 shim。安装这些 features 时，若依赖构建时记录的路径，应指定持久的 `--target-dir`；或者在清理构建产物前，把对应 shim 部署到可执行文件旁。厂商运行库依赖和 OpenVINO frontend/设备 plugin 也必须能被找到。`cargo install` 不会生成可直接搬到其他机器使用的 SDK/运行库分发包。配置见[后端部署说明](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr/README.md)和 [ORT 运行库说明](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-ort/README.md)；未准备全部 SDK 和运行库时，不要直接启用 `--all-features`。

### 参数与结果

| 参数 | 含义与默认值 |
| --- | --- |
| `--model PATH` | 模型文件，默认 `models/dfn3.onnx`。 |
| `--backend LIST` | 逗号分隔或重复指定，按给定顺序测量；默认 `gl,vk,dx`。 |
| `--warmup N` | 每个后端的额外预热帧数，不计入统计；默认 `100`，允许 `0`。 |
| `--frames N` | 每个后端正式测量的帧数；默认 `1000`，必须大于零。 |
| `--strength VALUE` | `0..=100` 的有限值，默认 `80`；另行报告后端实际强度。 |
| `--openvino-device DEVICE` | `CPU`、`GPU`、`GPU.n` 或 `NPU`；默认 `CPU`，只用于 OpenVINO。 |
| `--json` | stdout 输出 JSON 而非表格；进度和诊断仍写入 stderr。 |

单个后端失败后仍继续后面的条目，**任一后端失败，最终退出码为 1**。失败条目不编造耗时，因此退出非零时，JSON 里仍可能包含其他后端的有效结果。参数解析错误会在测量前退出。

初始化单独计时，不与预热、正式调用混算。正式测量继续使用预热后的状态和输入位置，不 reset。单帧耗时包含整个同步 `run`、主机/设备传输及完成等待，不只是 GPU kernel 时间；P50/P95/P99 使用 nearest-rank。FPS 与 RTF 按正式循环的总墙钟时间计算；`RTF < 1` 本身不保证实时任务不超时。这里不测设备 I/O、重采样、应用缓冲、模型算法延迟或主观音质。

JSON 保存配置、输入种子、OS/架构、debug-assertions 状态、各后端初始化/实际强度/统计值，以及 OpenVINO 实际执行设备。`cpu_fallback_allowed` 表示策略，不是实际 CPU 工作占比；明确选择 OpenVINO CPU 时该字段仍为 false。模型 hash、硬件、驱动和运行库/SDK 版本须另行记录，报告不会自动采集。完整公式、历史测量及源码复现步骤见[性能记录](https://github.com/lona-cn/anr/blob/v0.2.0/benchmark.md)。包内实现：[计时](src/benchmark.rs)、[统计](src/statistics.rs)、[报告](src/report.rs)。

### 许可与测试

项目代码采用 **MIT OR Apache-2.0**，任选其一：[LICENSE-MIT](LICENSE-MIT)、[LICENSE-APACHE](LICENSE-APACHE)。模型、驱动、SDK 和 Rust 依赖保留各自条款；再分发二进制前阅读[第三方分发说明](https://github.com/lona-cn/anr/blob/v0.2.0/crates/anr-cli/THIRD-PARTY-NOTICES.md)。

源码包包含统计/CLI 测试和自制的微型 ONNX 控制图，不包含完整 DFN3 模型。标为 ignored 的三 provider ORT 回归需要 Windows、对应 Cargo features 和匹配的 GPU/运行库；它验证 provider 行为，不证明 DFN3 音质或历史性能数值。普通测试不需要执行 GPU 推理。
