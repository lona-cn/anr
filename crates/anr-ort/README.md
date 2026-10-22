# anr-ort

[English](#english) | [简体中文](#简体中文)

## English

Stateful DeepFilterNet3 (DFN3) inference with ONNX Runtime: **48 kHz, mono `f32` audio**, processed in 480-sample frames. This crate provides the ORT session wrapper and model preparation APIs; it does not decode audio, resample, or download models. The [`anr` facade](../anr/README.md) adds model discovery and independent native backends. Requires the latest stable Rust toolchain.

### CPU quickstart

```toml
[dependencies]
anr-core = "0.1.1"
anr-ort = "0.1.1"
```

```rust,no_run
use anr_core::{NoiseReduction, Result};
use anr_ort::DFN3;

fn main() -> Result<()> {
    anr_ort::initialize_runtime()?;
    let environment = anr_ort::ort::init().build().map_err(anr_core::Error::backend)?;
    let mut reducer = DFN3::from_file(&environment, "models/dfn3.onnx")?;
    reducer.set_factor(80.0);

    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    reducer.run(DFN3::SAMPLE_RATE, &input, &mut output)?;
    reducer.reset();
    Ok(())
}
```

The model path is relative to the application's working directory. `run`, `factor`, `set_factor`, and `reset` are methods of `anr_core::NoiseReduction`; import the trait. The example processes one silent frame.

### Runtime and Cargo features

`anr_ort::ort` re-exports the exact runtime API used by this crate. `Cargo.lock` pins `ort` and `ort-sys` to `2.0.0-rc.15`; the workspace enables `std` and `api-27`, without `ndarray`. Two runtime versions serve different build paths:

- The pinned `ort-sys` downloader's distributions use ONNX Runtime **1.31.0**.
- The repository's application release configuration, `.github/onnxruntime.json`, pins Microsoft ONNX Runtime **1.28.0**. This is not the Cargo downloader's version.

A supplied library must expose API 27 and provide the operators/providers the application needs; a version label alone does not establish compatibility with a custom or minimal build.

| Feature | Effect |
| --- | --- |
| `download-binaries` (default) | Enables the upstream runtime downloader, shared-library copying, and native TLS. Requires an available distribution for the target. |
| `load-dynamic` | Loads an application-supplied shared library at runtime. |
| `cuda`, `tensorrt`, `directml`, `coreml`, `xnnpack` | Compiles the corresponding ORT provider support; does not select a device or install vendor runtimes/drivers. |

For a supplied runtime:

```toml
anr-ort = { version = "0.1.1", default-features = false, features = ["load-dynamic"] }
```

Set `ORT_DYLIB_PATH` to the shared-library **file**, then call `initialize_runtime()` before `ort::init().build()`. Unset or empty uses the platform-default name (`onnxruntime.dll`, `libonnxruntime.so`, or `libonnxruntime.dylib`). The helper returns loading failures, permits retries after failed loads, and reuses an already loaded library. In linked builds it is a no-op. It does not create an `Environment`; the application configures, builds, and retains that environment for session creation.

Cargo features are additive across a dependency graph. Disabling defaults on one dependency does not remove `download-binaries` enabled elsewhere. The facade's feature names use the `ort-` prefix; `ort-load-dynamic` and the provider features also enable its `ort` backend.

### Environment and session configuration

`DFN3::from_file(&environment, path)` is the CPU convenience constructor. It explicitly selects ORT's built-in CPU provider, disables environment provider lists and automatic accelerator selection, uses an independent pool with one intra-op thread, and disables intra/inter-op spinning. It works without device discovery in minimal runtimes.

For other settings, configure a `SessionBuilder`, use `session_from_file(builder, path, &providers)`, and pass the result to `DFN3::new(session)`. For example, with the `cuda` feature and its runtime dependencies:

```rust,no_run
use anr_core::NoiseReduction;
use anr_ort::{DFN3, ort, session_from_file};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    anr_ort::initialize_runtime()?;
    let environment = ort::init().build()?;
    let builder = ort::session::Session::builder(&environment)?
        .with_no_environment_execution_providers()?
        .with_independent_thread_pool()?
        .with_intra_threads(1)?
        .with_intra_op_spinning(false)?
        .with_inter_op_spinning(false)?
        .with_parallel_execution(false)?;
    let providers = [ort::ep::CUDA::default().build()];
    let session = session_from_file(builder, "models/dfn3.onnx", &providers)?;
    let mut reducer = DFN3::new(session)?;
    reducer.set_factor(80.0);
    Ok(())
}
```

For DirectML, also disable memory patterns with `with_memory_pattern(false)` and keep sequential execution. `session_from_file` preserves the builder's settings except for requested provider registration and the external-weight directory described below. Registration errors are returned rather than silently substituting a different provider. Ordinary per-operator CPU placement remains allowed: unsupported or CPU-preferred operators may run on CPU beside a GPU provider. `DFN3::new` preserves the session's placement policy.

Audio and state tensors are CPU-backed even for GPU sessions; ORT handles transfers. Buffers are allocated during construction, with two alternating state buffers rather than a per-frame state copy. This is a synchronous API, not a guarantee of audio-callback safety.

### In-memory model preparation

`session_from_file` prepares CUDA, TensorRT, and DirectML graphs before session creation. If several are requested, target selection is TensorRT, then DirectML, then CUDA. Other providers retain upstream file loading.

CUDA expands `FusedConv(Sigmoid)` while retaining its supported ReLU and matrix fusions. TensorRT/DirectML also expand `FusedConv(Relu)` and eligible non-transposed `FusedMatMul`, preserving `alpha`; scaled-matmul expansion requires opset 13 or newer.

All three targets lower statically indexed state sequences and proven-safe fixed-shape tensor conditionals. `float32` `NonZero` used only for an all-zero predicate becomes a fixed-shape reduction. Proven uniform boolean selections and stateless dead dependencies are removed; unknown or stateful computations keep their execution dependencies. Runtime conditions and attenuation remain live. Unproven patterns remain unchanged.

Preparation happens once per session, changes neither source files nor weights, and creates no derived ONNX file or disk cache. Relative external weights resolve beside the source model unless the builder already specifies `session.model_external_initializers_file_folder_path`. This prepared-file path requires a UTF-8 model directory.

Callers with bytes can use the public API directly:

```rust,no_run
use anr_ort::{ModelTarget, ort, prepare_model};

fn load_bytes(
    mut builder: ort::session::builder::SessionBuilder,
    bytes: &[u8],
) -> Result<ort::session::Session, Box<dyn std::error::Error>> {
    let prepared = prepare_model(bytes, ModelTarget::DirectML)?;
    Ok(builder.commit_from_memory(prepared.as_ref())?)
}
```

`prepare_model(&[u8], ModelTarget) -> anr_core::Result<Cow<'_, [u8]>>` borrows unchanged input and owns rewritten bytes. Available targets are `Cuda`, `TensorRT`, and `DirectML`. It does not register providers or configure session options; the example assumes the caller has done that. In-memory callers must configure external-weight resolution themselves. Direct upstream `commit_from_file` bypasses preparation; `DFN3::new` cannot alter an already-loaded graph.

### Model port contract

`DFN3::new(session)` validates exactly three inputs and three outputs, all `f32`. Names, not port order, identify tensors. `[]` is scalar and `[1]` is a one-element vector.

| Direction | Name | Runtime shape |
| --- | --- | --- |
| Input | `input_frame` | `[480]` |
| Input | `states` | `[45304]` |
| Input | `atten_lim_db` | `[]` or `[1]` |
| Output | `enhanced_audio_frame` | `[480]` |
| Output | `new_states` | `[45304]` |
| Output | `lsnr` | `[]` or `[1]` |

One-dimensional symbolic lengths are accepted in declarations; execution still requires these fixed audio/state lengths. Constants are `DFN3::SAMPLE_RATE = 48000`, `FRAME_SIZE = 480`, and `STATE_SIZE = 45304`. ORT checks actual audio/state outputs against preallocated buffers. `lsnr` is checked only in the declaration and is not requested or returned.

Unlike the native compute implementations, this crate does not require one file hash. The graph must nevertheless use the expected state ordering and attenuation semantics; matching shapes alone is insufficient. The upstream multi-file DeepFilterNet export is not this single-frame interface.

### Streaming, errors, and strength

- Keep one reducer per continuous channel/stream. `reset()` clears state and preserves strength.
- `run` accepts any nonzero input length at 48,000 Hz with equal-length output. Each short final frame is zero-padded to 480 samples; padding advances a full frame of state, while only requested samples are returned. Prefer multiples of 480 between calls.
- There is no model-latency compensation or end-of-stream flush. Audio decoding, channel separation, and resampling belong to the application.
- Empty input, wrong rate, and unequal lengths return shared validation errors without changing output/state. Inference failures may leave earlier output written and state partly advanced. Reset before reuse; recreate if the runtime/session is no longer usable. See [`anr-core`](../anr-core/README.md) for error variants.

Initial attenuation is **3 dB** (`factor() == 5`). `set_factor` clamps to `0..=100` and sets `round(factor / 100 * 60)` dB. `factor()` converts the actual dB setting back to the percentage scale and rounds, so readback may differ from the supplied value. `set_atten_lim_db` / `get_atten_lim_db` access `0..=60` dB directly and preserve fractional dB. Both setters ignore NaN and saturate infinities at the nearest bound.

Strength is an attenuation limit, not output gain or a dry/wet mix. Zero still runs the graph and is not an unchanged-audio bypass. The dB methods belong only to this concrete ORT type, not `NoiseReduction` or the compute APIs.

### License

MIT OR Apache-2.0, at your option: [LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE). Model weights, ONNX Runtime, and provider dependencies have separate terms. ONNX parsing uses unmodified `onnx-protobuf` 0.2.3 (MPL-2.0) and `protobuf` 3.4.0 (MIT); see the [third-party notices](../anr-cli/THIRD-PARTY-NOTICES.md) for source and distribution requirements.

## 简体中文

`anr-ort` 使用 ONNX Runtime 执行有状态的 DeepFilterNet3（DFN3）推理：**48 kHz、单声道 `f32` 音频**，每帧 480 个采样。本包负责 ORT Session 封装和模型准备，不解码音频、不重采样，也不下载模型。[`anr` facade](../anr/README.md) 另提供模型查找和独立原生后端。需要最新 Rust stable 工具链。

### CPU 入门

```toml
[dependencies]
anr-core = "0.1.1"
anr-ort = "0.1.1"
```

```rust,no_run
use anr_core::{NoiseReduction, Result};
use anr_ort::DFN3;

fn main() -> Result<()> {
    anr_ort::initialize_runtime()?;
    let environment = anr_ort::ort::init().build().map_err(anr_core::Error::backend)?;
    let mut reducer = DFN3::from_file(&environment, "models/dfn3.onnx")?;
    reducer.set_factor(80.0);
    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    reducer.run(DFN3::SAMPLE_RATE, &input, &mut output)?;
    reducer.reset();
    Ok(())
}
```

模型路径相对于应用的工作目录。`run`、`factor`、`set_factor` 和 `reset` 来自 `anr_core::NoiseReduction`，需引入 trait。示例处理一帧静音。

### 运行库与 Cargo features

`anr_ort::ort` 重导出本包使用的运行库 API。`Cargo.lock` 固定 `ort`、`ort-sys` 为 `2.0.0-rc.15`；工作区启用 `std`、`api-27`，不启用 `ndarray`。两条构建路径使用不同版本：

- 此版 `ort-sys` 下载器使用 ONNX Runtime **1.31.0** 分发包。
- 应用发布配置 `.github/onnxruntime.json` 固定 Microsoft ONNX Runtime **1.28.0**，不是 Cargo 下载器的版本。

自备库必须提供 API 27 以及所需算子/provider；版本号本身不能保证自定义或 minimal 构建兼容。

| Feature | 作用 |
| --- | --- |
| `download-binaries`（默认） | 上游运行库下载、共享库复制和 native TLS；目标平台须有可用分发包。 |
| `load-dynamic` | 在运行时加载应用提供的共享库。 |
| `cuda`、`tensorrt`、`directml`、`coreml`、`xnnpack` | 编入对应 ORT provider 支持，不自动选设备或安装厂商运行库/驱动。 |

使用自备运行库：

```toml
anr-ort = { version = "0.1.1", default-features = false, features = ["load-dynamic"] }
```

将 `ORT_DYLIB_PATH` 设为共享库**文件**，在 `ort::init().build()` 前调用 `initialize_runtime()`。未设置或为空时使用平台默认库名：`onnxruntime.dll`、`libonnxruntime.so` 或 `libonnxruntime.dylib`。该函数正常返回加载错误，失败后可重试，已加载的库会复用；链接构建中不执行操作。它不创建 `Environment`，环境由应用配置、创建并持有，供 Session 使用。

Cargo features 在依赖图中取并集：单个依赖关闭 defaults，不能撤销其他依赖启用的 `download-binaries`。facade 使用 `ort-` 前缀；其 `ort-load-dynamic` 和 provider features 还会启用 `ort` 后端。

### Environment 与 Session 配置

`DFN3::from_file(&environment, path)` 是 CPU 便捷构造器：显式选择内置 CPU provider，关闭环境 provider 列表和自动加速器选择，使用独立线程池、一个 intra-op 线程，禁用 intra/inter-op 忙等待；minimal 运行库不需要设备发现 API。

需要其他配置时，先配置 `SessionBuilder`，通过 `session_from_file(builder, path, &providers)` 创建 Session，再交给 `DFN3::new(session)`。上方英文示例给出了完整 CUDA 配置，需要启用 `cuda` 并准备相应运行库。DirectML 还需 `with_memory_pattern(false)`，并关闭并行执行。加载器保留 builder 的其他设置，只注册指定 provider，并在需要时补充外部权重目录。

显式 provider 注册失败会报错，不会静默改用另一 provider。正常的按算子 CPU 分配仍然允许：GPU provider 不支持或更适合 CPU 的算子可在 CPU 上运行。`DFN3::new` 保留 Session 的分配策略。

音频和状态张量始终位于 CPU 内存，由 ORT 处理 GPU 传输。构造时分配缓冲区，以两份状态缓冲交替读写，避免逐帧复制整份状态。这是同步 API，不保证音频回调安全。

### 内存中的模型准备

`session_from_file` 会在创建 CUDA、TensorRT、DirectML Session 前准备模型；同时请求多个时，目标优先级为 TensorRT、DirectML、CUDA。其他 provider 沿用上游文件加载。

CUDA 展开 `FusedConv(Sigmoid)`，保留其支持的 ReLU 卷积和矩阵融合。TensorRT/DirectML 另展开 `FusedConv(Relu)` 和符合条件的无转置 `FusedMatMul`，保留 `alpha`；带缩放矩阵乘法的展开要求 opset 至少为 13。

三个目标都会转换常量索引的状态序列，以及已证明可安全提前执行的固定形状张量分支。仅用于全零判断的 `float32` `NonZero` 改为固定形状归约；消除已证明恒定的布尔选择及无状态死计算，保留未知或有状态计算的执行依赖。运行时条件和衰减量仍动态生效，无法证明适用的模式保持原样。

每个 Session 只准备一次，不改写源文件或权重，不生成派生 ONNX 文件或磁盘缓存。外部权重默认相对于源模型目录解析；builder 已配置 `session.model_external_initializers_file_folder_path` 时不会覆盖。此准备路径要求模型目录为有效 UTF-8。

已有模型字节时可调用 `prepare_model(&[u8], ModelTarget) -> anr_core::Result<Cow<'_, [u8]>>`，再通过 builder 的 `commit_from_memory(prepared.as_ref())` 加载。`ModelTarget` 包括 `Cuda`、`TensorRT`、`DirectML`；未改变时借用输入字节，改变时返回拥有的缓冲区。该 API 不注册 provider 或配置 Session，外部权重目录也须由调用方设置。直接调用上游 `commit_from_file` 会绕过准备；`DFN3::new` 无法修改已加载的图。

### 模型端口契约

`DFN3::new(session)` 要求恰好三个输入、三个输出，全部为 `f32`，按名称而非顺序匹配。`[]` 为标量，`[1]` 为单元素向量。

| 方向 | 名称 | 运行时形状 |
| --- | --- | --- |
| 输入 | `input_frame` | `[480]` |
| 输入 | `states` | `[45304]` |
| 输入 | `atten_lim_db` | `[]` 或 `[1]` |
| 输出 | `enhanced_audio_frame` | `[480]` |
| 输出 | `new_states` | `[45304]` |
| 输出 | `lsnr` | `[]` 或 `[1]` |

声明允许一维符号长度，运行时仍须满足上述音频/状态长度。常量为 `DFN3::SAMPLE_RATE = 48000`、`FRAME_SIZE = 480`、`STATE_SIZE = 45304`。ORT 根据预分配缓冲区检查实际音频/状态输出；`lsnr` 仅检查声明，不请求或返回其值。

本包不像原生 Compute 那样限定一个文件摘要，但模型必须采用预期的状态顺序和衰减语义，端口形状相同不足以保证兼容。上游多文件 DeepFilterNet 导出不符合此单帧接口。

### 流、错误与强度

- 每路连续音频/声道使用独立实例；`reset()` 清除状态，保留强度。
- `run` 接受任意非零长度的 48,000 Hz 输入，输出必须等长。尾帧补零至 480 采样，状态推进完整一帧，只返回请求长度；连续调用尽量使用 480 的整数倍。
- 不补偿模型延迟，不提供流结束 flush。解码、声道拆分和重采样由应用负责。
- 空输入、错误采样率和长度不等的校验错误不会改变输出/状态。推理错误可能发生在已有输出写入、状态推进之后；再次使用前 reset，运行库/Session 无法恢复时重新创建。错误类型见 [`anr-core`](../anr-core/README.md)。

初始衰减为 **3 dB**（`factor() == 5`）。`set_factor` 将值限制到 `0..=100`，再按 `round(factor / 100 * 60)` 设置整分贝；`factor()` 从实际分贝换算回百分比并取整，读回值可能不同。`set_atten_lim_db` / `get_atten_lim_db` 直接访问 `0..=60` dB，保留小数。两种 setter 都忽略 NaN，正负无穷分别饱和到上限和下限。

强度是衰减上限，不是输出增益或干湿混合比例。零仍执行模型，不表示原音旁路。分贝方法仅属于 ORT 具体类型，不属于 `NoiseReduction` 或 Compute API。

### 许可

MIT OR Apache-2.0，任选其一：[LICENSE-MIT](LICENSE-MIT)、[LICENSE-APACHE](LICENSE-APACHE)。模型权重、ONNX Runtime 和 provider 依赖另有许可条件。ONNX 解析使用未修改的 `onnx-protobuf` 0.2.3（MPL-2.0）和 `protobuf` 3.4.0（MIT）；源码和分发要求见[第三方说明](../anr-cli/THIRD-PARTY-NOTICES.md)。
