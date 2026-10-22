# anr-cli

[English](#english) | [简体中文](#简体中文)

## English

`anr-cli` provides the `anr` command for DFN3 audio denoising: individual files, serial batches, and stdin/stdout streams. It handles decoding, band-limited resampling, and independent channel state. ONNX Runtime (ORT) is the default backend; nine native backends are opt-in. Audio processing needs neither FFmpeg nor Python. This is a file/pipe tool, not an audio-device driver or virtual microphone.

### Installation

#### Application ZIP

The [release workflow](../../.github/workflows/ci.yml) packages Windows x64, Linux x64, macOS arm64, and macOS x64 applications as `anr-<version-or-commit>-<target>.zip`. Each package contains the executable, CPU ONNX Runtime **1.28.0**, `models/dfn3.onnx`, and licenses/notices. The separate [model release](https://github.com/lona-cn/anr/releases/tag/model-dfn3-fe5eb64fa2e4) contains model assets, not the CLI.

When using a packaged build, verify its attached `SHA256SUMS`, extract the entire archive, and run:

```sh
./anr denoise input.wav --output clean.wav
```

On Windows, use `.\anr.exe`. Keep the runtime libraries and `models/` beside the executable; discovery does not depend on the working directory. An existing `ORT_DYLIB_PATH` overrides the bundled runtime, so unset it to use the package's copy.

- Windows may require the [Microsoft Visual C++ x64 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
- Linux packages are glibc builds produced on Ubuntu 24.04, not musl builds or a promise of compatibility with older glibc.
- macOS packages use ad-hoc signing, not notarization; follow the system's approval procedure for downloaded applications.
- These packages contain CPU ORT, not GPU providers or native vendor SDKs. Packaging configuration is not a record of platform test results; see [RELEASING.md](../../RELEASING.md) for the release procedure.

#### Source installation

Use the latest stable Rust toolchain. From a source checkout's workspace root:

```text
cargo install --path crates/anr-cli --locked
anr --help
anr backends --json
```

The executable is named `anr`, not `anr-cli`; put Cargo's installation directory on `PATH`. Source builds need the platform linker/toolchain: MSVC C++ tools and Windows SDK on Windows, C/C++ linking tools plus `pkg-config` and OpenSSL development files (such as `libssl-dev`) on Linux, or Xcode Command Line Tools on macOS. Default features download a prebuilt CPU ORT runtime and prepare the model, so a first build normally needs network access.

**Intel macOS needs a supplied or source-built ORT runtime**, rather than assuming a prebuilt runtime is available. For example, after obtaining an API-27-compatible CPU shared library (including ONNX Runtime 1.28.0):

```sh
export ORT_DYLIB_PATH="/absolute/path/to/libonnxruntime.dylib"
cargo install --path crates/anr-cli --locked --no-default-features --features ort-load-dynamic,download-models,copy-models
anr backends --json
```

Use the actual library **file**, and retain the setting when running `anr`. Building ORT itself requires its own tools; see the [upstream build instructions](https://onnxruntime.ai/docs/build/inferencing.html). Python/CMake used for that build are not installed-CLI audio dependencies.

### Commands and model selection

```text
anr denoise voice.wav --output clean.wav
anr denoise a.flac b.mp3 --output-dir cleaned
anr inspect voice.wav
anr inspect voice.wav --json
anr backends --json
anr denoise --help
```

`inspect` probes the selected audio track without loading a model or decoding the entire recording. JSON fields are `format`, `codec`, `track_id`, `sample_rate`, `channels`, `frames`, and `duration_seconds`; unknown counts/durations are `null`, and known duration is header-derived. It accepts encoded input, not a raw-input specification.

`backends` lists all ten backend names and their compiled status. ORT additionally lists `cpu`, `cuda`, `tensorrt`, `directml`, `coreml`, and `xnnpack`. Every `runtime_available` field is `null`: the command does not load runtimes, create devices, or test inference.

`--model PATH` is optional. An explicit path wins and never falls back on failure; relative paths resolve from the working directory. Otherwise, the CLI uses `models/dfn3.onnx` beside the executable, then the model source recorded at build time. A bad bundled file is an error, not a reason to try a different model. No model is downloaded at runtime.

Default features are `ort-download-binaries`, `download-models`, and `copy-models`:

- `download-models` downloads and verifies the pinned model in `$CARGO_HOME/anr/models/dfn3-fe5eb64fa2e4/` (`~/.cargo` by default).
- `copy-models` copies the prepared model into `models/dfn3.onnx` in Cargo's profile, examples, and test-output directories.
- Set `ANR_MODEL_PATH` to a local model **before building** to bypass downloading; `copy-models` alone requires this setting. Use an absolute path for reproducible builds. A cold cache with `CARGO_NET_OFFLINE=true` fails instead of downloading.
- `cargo install` can use the persistent build-time model source. For relocation to another machine, ship the model beside the executable with its licenses/notices; the build machine's cache is not portable.

Disabling default features removes automatic model preparation as well as default ORT support. See the [library reference](../anr/README.md) for model preparation and the [ORT reference](../anr-ort/README.md) for the accepted model port contract. Native backends require this exact original model SHA-256:

```text
fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9
```

Matching tensor shapes alone does not make another export compatible with native backends.

### Denoise arguments

```text
anr denoise INPUT... --output PATH [OPTIONS]
anr denoise INPUT... --output-dir DIR [OPTIONS]
```

| Argument | Contract and default |
| --- | --- |
| `INPUT...` | One or more explicit file paths. `-` means stdin and requires a single input with `--output`. No directory traversal or program-side glob expansion. |
| `-o, --output PATH` | One input's destination; `-` means stdout. Mutually exclusive with `--output-dir`. |
| `--output-dir DIR` | Creates the directory if needed; serially writes `stem.wav`, or `stem.f32le` for explicit raw output. Colliding stems are errors, not automatically renamed. |
| `--model PATH` | Optional override; discovery order is described above. `--model -` names a real file, not stdin. |
| `--backend ort\|gl\|gles\|vk\|wgpu\|dx\|cuda\|tensorrt\|openvino\|opencl` | Default `ort`, even in a native-only build. Select a compiled backend explicitly. |
| `--provider cpu\|cuda\|tensorrt\|directml\|coreml\|xnnpack` | ORT only; omitted means `cpu`. Any explicit provider with a native backend is an argument error. |
| `--openvino-device DEVICE` | OpenVINO only: `CPU` (default), `GPU`, `GPU.n` such as `GPU.0`, or `NPU`. No AUTO/HETERO or CPU fallback from a requested GPU/NPU. |
| `--strength VALUE` | Default `80`; finite `0..=100`, without clamping invalid arguments. Zero selects DFN3's attenuation branch, not a raw-audio bypass. |
| `--input-format auto\|f32le` | Default `auto`: probe encoded audio. Raw PCM requires explicit `f32le`. |
| `--input-rate HZ` / `--input-channels N` | Required positive values for raw input (`u32` rate, `u16` channels). Invalid for `auto`; cannot override an audio header. |
| `--output-format wav\|f32le` | File extensions `.wav` / `.f32le` infer format, case-insensitively; other extensions need this option. Directory output defaults to WAV. Stdout requires explicit `f32le`. |
| `--output-rate HZ` | Positive `u32`; defaults to the input rate. |
| `--wav-format pcm16\|pcm24\|float32` | Default `pcm16`. Explicit use with raw output is an error. |
| `--overwrite` | Replaces an existing destination only after successful processing. Never permits replacing an input, the selected model, or a result already saved in this batch. |
| `-q, --quiet` / `-v, --verbose` | Mutually exclusive global options. Quiet keeps errors; verbose adds processing specifications. Diagnostics and completion statistics go to stderr. |
| `-h, --help` / `-V, --version` | Display help / top-level version and exit. |

Global argument and destination checks, runtime/provider setup, and the first model instance are completed before output creation. The output directory is checked again after creation. Input/model paths and file identities protect against aliases, symlinks, hard links, and case-insensitive collisions; checks repeat before each file and before commit. A later destination that aliases an earlier completed result fails rather than replacing it.

Each channel has its own engine. Engines are reused where possible and reset between files. Decode, inference, or output failure for one file removes its temporary output, reports the error, and continues with the remaining files. Global initialization errors, failures to initialize additional channel engines, cancellation, and broken pipes stop the batch. Successful files remain; there is no batch rollback. If a saved file's identity cannot be verified, the CLI reports that it was saved and stops.

File output uses a temporary file in the destination directory. Only after inference, encoding, and WAV finalization succeed is it committed to the final path; the default commit does not clobber a destination created concurrently. `--overwrite` does not truncate the old file in advance. This is not a power-loss durability guarantee. `--output-dir` creates parents; with `--output`, the parent directory must already exist.

### Formats, duration, and latency

Symphonia 0.5.5 with `all` features handles container probing and decoding, including WAV/PCM, FLAC, MP3, Ogg/Vorbis, and AAC-LC/M4A. Codec/profile support still matters: recognizing a container does not imply support for every track inside it; Opus is not in the current codec set. The first track with a supported decoder is selected. Changes of track, sample rate, or channel layout during processing are rejected.

Encoded stdin is non-seekable: a container that requires seeking may fail. The CLI neither buffers the whole stream nor silently spools it to disk. Reliable declared frame counts in fixed-width WAV are checked for premature EOF; unknown-length WAV markers remain streamable. Compressed-format duration estimates are not exact integrity budgets, and `inspect` is not a full-file integrity check.

Output is WAV (PCM16, PCM24, or float32) or raw f32le; there are no MP3/AAC encoders. WAV retains samples and necessary headers, not source tags, artwork, chapters, or special channel-layout metadata. Integer samples are rounded and clamped; non-finite input/output samples are errors. WAV output is limited by RIFF's 32-bit length fields, not RF64.

The pipeline preserves channel count/order without mixing. It uses bounded streaming buffers, resamples to DFN3's **48 kHz / 480-sample** frames, and resamples to the requested output rate. Equal-rate stages bypass resampling. Samples accumulate across decoder packets; only the last model frame is zero-padded at EOF, and padding is trimmed.

Output length is based on **actually decoded frames**, not a header estimate: at the original rate, exactly the input frame count; with a rate change, `round(input_frames × output_rate / input_rate)` (half up). Here a frame is one sample per channel at the same instant. Resampler filter delay is compensated and its tail drained. **DFN3's own algorithmic latency is not compensated**, and the backend exposes no model end-of-stream flush. Equal length does not imply latency-free processing. See [DESIGN.md](DESIGN.md) for the stream architecture.

### Binary-safe raw streams

Raw audio is interleaved, little-endian IEEE 754 `f32`, without a header. A frame of N channels occupies `4 × N` bytes. Incomplete samples/channel frames at EOF are errors. Stdin and audio stdout must not be interactive terminals. Only raw f32le is supported on stdout; WAV needs seeking to finalize its header.

Unix shell example (uses automatic model discovery):

```sh
anr denoise - --input-format f32le --input-rate 48000 --input-channels 1 --output - --output-format f32le < input.f32le > clean.f32le
```

Windows `cmd.exe`, with an explicit model:

```bat
anr denoise - --input-format f32le --input-rate 48000 --input-channels 1 --output - --output-format f32le --model "C:\models\dfn3.onnx" < input.f32le > clean.f32le
```

Keep stderr separate—never use `2>&1` for an audio stream. PowerShell 5.1 text pipelines and `Get-Content` / `Set-Content` are not binary-safe. PowerShell 7.4+ supports native byte pipelines and native stdout redirection; see the [PowerShell redirection reference](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_redirection?view=powershell-7.5).

A caller can pass binary handles directly without shell conversion (Python is optional):

```python
import subprocess

with open("input.f32le", "rb") as source, open("clean.f32le", "wb") as target:
    subprocess.run([
        "anr", "denoise", "-", "--input-format", "f32le",
        "--input-rate", "48000", "--input-channels", "1",
        "--output", "-", "--output-format", "f32le",
    ], stdin=source, stdout=target, check=True)
```

Redirection or the caller opens/truncates `clean.f32le` itself; that file is **not** protected by the CLI's output transaction or `--overwrite` checks. Audio already sent to stdout cannot be rolled back. Check exit status and stderr before accepting a stream as complete.

| Exit code | Meaning |
| --- | --- |
| `0` | Success, help/version, or `BrokenPipe` when a downstream reader closes stdout early. A broken-pipe success does not mean the full recording was produced. |
| `1` | Runtime failure, including any failed files in a completed batch. |
| `2` | Argument/usage error. |
| `130` | Ctrl+C cancellation. |

Cancellation unwinds the processing loop and removes the current temporary file; previously committed files remain. Blocked stdin/stdout does not require EOF before cancellation can be handled. Cancellation is cooperative between operations, not a promise to interrupt an in-progress vendor inference call.

### Backend builds and deployment

| Build feature | Runtime selection | Requirements |
| --- | --- | --- |
| `ort-download-binaries` (default) | `--backend ort --provider cpu` | Downloads the prebuilt CPU runtime at build time where available. |
| `ort` | `--backend ort` | A supplied runtime, configured with `ORT_LIB_LOCATION` for linking. |
| `ort-load-dynamic` | `--backend ort` | API-27-compatible shared library; `ORT_DYLIB_PATH` points to its file. No ORT downloader/TLS dependency when defaults are disabled. |
| `ort-cuda`, `ort-tensorrt`, `ort-directml`, `ort-coreml`, `ort-xnnpack` | `--provider cuda`, `tensorrt`, `directml`, `coreml`, `xnnpack` | Matching ORT runtime, platform, provider libraries and drivers. DirectML is Windows-only; CoreML is macOS-only. |
| `gl` | `--backend gl` | Hardware desktop GL 4.3: Windows or Linux EGL/pbuffer. |
| `gles` | `--backend gles` | Real native ES 3.1+: Windows WGL or Linux hardware EGL/pbuffer, not desktop-GL ES compatibility. No Android/WebGL/macOS context creation. |
| `vk` | `--backend vk` | Windows/Linux hardware Vulkan 1.1, independent of wgpu. |
| `wgpu` | `--backend wgpu` | Native hardware Vulkan/DX12/Metal adapter on Windows/Linux/macOS. |
| `dx` | `--backend dx` | Windows hardware D3D12 and system D3DCompiler 47; not DirectML. |
| `cuda` | `--backend cuda` | NVIDIA Driver API 12.0+ and NVRTC 12/13 (13 preferred); driver must also support the generated PTX. No build-time CUDA SDK required. |
| `tensorrt` | `--backend tensorrt` | TensorRT/CUDA SDKs and C++17 compiler at build time; matching runtime libraries and NVIDIA driver at runtime. |
| `openvino` | `--backend openvino [--openvino-device CPU]` | OpenVINO C++ SDK and C++17 compiler at build time; Runtime, ONNX frontend, device plugin and dependencies at runtime. |
| `opencl` | `--backend opencl` | OpenCL ICD loader plus hardware GPU vendor runtime/driver with OpenCL C 1.2+, online compiler, and required FP32/device capabilities. No build-time SDK required. |

Features add support; they do not install vendor SDKs or drivers. ORT permits per-operator CPU placement alongside GPU providers; explicit provider registration/initialization failures remain errors. Native GPU backends do not select software/CPU fallback. OpenVINO's default is explicitly CPU; requested GPU/NPU failures remain errors, and NPU arithmetic is device-dependent. CUDA/TensorRT/DirectML ORT sessions prepare the graph in memory through `anr-ort`, without changing the source model or requiring Python; see the [ORT reference](../anr-ort/README.md). GPU in-memory loading currently requires a UTF-8 model directory for external-weight resolution.

Known OpenVINO GPU compatibility issue: compilation of the original graph has failed at `NonZero_1_count` / `count_nonzero` kernel selection. Selecting a GPU does not bypass unsupported operators; NPU support likewise depends on the installed plugin and device.

```text
cargo build -p anr-cli --release
cargo build -p anr-cli --features ort-directml
cargo build -p anr-cli --no-default-features
cargo run -p anr-cli --release --no-default-features --features gles -- denoise input.wav --model /path/to/dfn3.onnx --backend gles --output clean.wav
```

A no-feature build can inspect audio and list backends, but cannot denoise. Native features do not enable ORT; use `--no-default-features` to omit it. Native-only builds must pass `--backend` because the argument default remains `ort`.

For SDK builds, set real paths before building (PowerShell example):

```powershell
$env:TENSORRT_ROOT = "C:\sdk\TensorRT"
$env:CUDA_ROOT = "C:\sdk\CUDA"
$env:OPENVINO_ROOT = "C:\sdk\openvino"
cargo build -p anr-cli --release --no-default-features --features cuda,tensorrt,openvino,opencl
```

TensorRT accepts `CUDA_PATH` instead of `CUDA_ROOT`, with `TENSORRT_LIB_DIR` / `CUDA_LIB_DIR` overrides for link-library directories. OpenVINO also accepts `OpenVINO_DIR` within its SDK/CMake layout or the official wheel's `openvino` directory containing `include/` and `libs/`; Python is not a runtime dependency of the resulting CLI.

For custom deployment:

- CUDA loads `nvrtc64_130_0.dll` / `nvrtc64_120_0.dll` on Windows or `libnvrtc.so.13` / `.12` / unversioned on Linux. Put matching NVRTC/builtins libraries on `PATH` or `LD_LIBRARY_PATH`. Driver API 12.0 alone does not guarantee NVRTC 13 PTX compatibility.
- TensorRT needs the build's `OUT_DIR/anr_tensorrt_native.dll` (Linux: `libanr_tensorrt_native.so`). `ANR_TENSORRT_SHIM` overrides the shim file. On Windows only, `ANR_TENSORRT_RUNTIME` accepts a TensorRT library file or DLL directory; Linux ignores this variable and needs libraries discoverable through platform loader paths such as rpath or `LD_LIBRARY_PATH`. Deploy compatible TensorRT/CUDA libraries as well. There is no persistent engine cache, so engine construction affects cold startup.
- OpenVINO needs `anr_openvino_native.dll` (`libanr_openvino_native.so` / `.dylib`) from `OUT_DIR`. `ANR_OPENVINO_SHIM` overrides the shim file; `ANR_OPENVINO_RUNTIME` is a Runtime library **file**, not a directory. Preserve the runtime/frontend/plugin layout and configure the platform library search path.
- Place shims beside the executable or set their overrides. Cargo installation does not collect the private shims or vendor libraries. Do not depend on SDK paths recorded from another build machine.
- On Windows, TensorRT searches imported DLL names in the shim directory, executable directory, existing absolute `PATH` entries, then recorded SDK directories, deduplicated in that order. Explicit `ANR_TENSORRT_RUNTIME` and `CUDA_ROOT` / `CUDA_PATH` take precedence; invalid overrides fail rather than selecting an unrelated installation. Empty/relative `PATH` entries do not enable current-directory search; the loader does not rewrite `PATH`.
- Shared ORT linking may need `ORT_PREFER_DYNAMIC_LINK=1` and the platform library search path in addition to `ORT_LIB_LOCATION`. Dynamic loading failures return exit code 1 before output creation.

Hardware-dependent tests are opt-in: set `ANR_DFN3_MODEL` and use `cargo test -p anr-cli --release --no-default-features --features gl,gles,vk,wgpu,dx,cuda,opencl --test cli compute_ -- --include-ignored`, selecting only locally supported features and adding SDK features only with their SDKs configured. These tests cover stream/file behavior, not perceptual quality or universal device compatibility. [benchmark.md](../../benchmark.md) records performance measurements separately.

### License

Project code is **MIT OR Apache-2.0**: [LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE). Model provenance and the supplied asset's authorization are documented in the [model notice](../../.github/models/NOTICE.md). Dependencies, runtimes, models, and SDKs retain their own terms; redistribution and source-availability obligations are collected in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

## 简体中文

`anr-cli` 提供 `anr` 命令，用于 DFN3 音频降噪：单文件、串行批处理和 stdin/stdout 流。CLI 负责解码、带限重采样和声道独立状态，默认使用 ONNX Runtime（ORT），另有九种可选原生后端。处理音频不需要 FFmpeg 或 Python。本工具不提供音频设备驱动或虚拟麦克风。

### 安装

#### 应用 ZIP

[发布工作流](../../.github/workflows/ci.yml) 为 Windows x64、Linux x64、macOS arm64 和 macOS x64 打包 `anr-<版本或提交>-<target>.zip`，包含可执行文件、CPU ONNX Runtime **1.28.0**、`models/dfn3.onnx` 和许可/通知。独立的[模型 Release](https://github.com/lona-cn/anr/releases/tag/model-dfn3-fe5eb64fa2e4) 只提供模型资产，不是 CLI。

使用打包版本时，先核对附带的 `SHA256SUMS`，完整解压后运行：

```sh
./anr denoise input.wav --output clean.wav
```

Windows 使用 `.\anr.exe`。运行库和 `models/` 应保留在可执行文件旁，自动查找不依赖当前工作目录。已有 `ORT_DYLIB_PATH` 会覆盖随包运行库；使用随包版本时应移除该设置。

- Windows 可能需要 [Microsoft Visual C++ x64 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)。
- Linux 包在 Ubuntu 24.04 上构建，使用 glibc，不是 musl 包，也不保证兼容更旧的 glibc。
- macOS 包使用 ad-hoc 签名，未公证；下载后按系统要求完成允许打开的操作。
- 包内只有 CPU ORT，不包含 GPU provider 或原生厂商 SDK。打包配置不等于平台测试结果；发布流程见 [RELEASING.md](../../RELEASING.md)。

#### 从源码安装

使用最新 Rust stable，在源码工作区根目录执行：

```text
cargo install --path crates/anr-cli --locked
anr --help
anr backends --json
```

二进制名为 `anr`，不是 `anr-cli`；将 Cargo 安装目录加入 `PATH`。构建依赖平台工具链：Windows 需要 MSVC C++ 工具和 Windows SDK；Linux 需要 C/C++ 链接工具、`pkg-config` 和 OpenSSL 开发文件（例如 `libssl-dev`）；macOS 需要 Xcode Command Line Tools。默认 features 下载预编译 CPU ORT 并准备模型，首次构建通常需要联网。

**Intel macOS 需要自备或源码构建 ORT 运行库**，不能假定有预编译版本。准备好兼容 API 27 的 CPU 共享库（包括 ONNX Runtime 1.28.0）后，例如：

```sh
export ORT_DYLIB_PATH="/absolute/path/to/libonnxruntime.dylib"
cargo install --path crates/anr-cli --locked --no-default-features --features ort-load-dynamic,download-models,copy-models
anr backends --json
```

路径必须指向实际的库**文件**，后续运行时仍需保留设置。自行构建 ORT 的工具要求见[上游说明](https://onnxruntime.ai/docs/build/inferencing.html)；其中的 Python/CMake 不是安装后 CLI 处理音频的依赖。

### 命令与模型选择

```text
anr denoise voice.wav --output clean.wav
anr denoise a.flac b.mp3 --output-dir cleaned
anr inspect voice.wav
anr inspect voice.wav --json
anr backends --json
anr denoise --help
```

`inspect` 探测所选音轨，不加载模型，也不解码整段录音。JSON 字段为 `format`、`codec`、`track_id`、`sample_rate`、`channels`、`frames`、`duration_seconds`；未知帧数/时长为 `null`，已知时长来自文件头。它接受带头输入，不提供 raw 规格参数。

`backends` 列出十种后端及其编译状态；ORT 另列 `cpu`、`cuda`、`tensorrt`、`directml`、`coreml`、`xnnpack`。所有 `runtime_available` 均为 `null`：该命令不加载运行库、不创建设备，也不执行推理。

`--model PATH` 可省略。显式路径优先且失败不回退，相对路径以当前工作目录为准。省略时先用可执行文件旁的 `models/dfn3.onnx`，再用构建时记录的模型源路径；随包模型无效会报错，不改试另一模型。运行时不下载模型。

默认 features 为 `ort-download-binaries`、`download-models`、`copy-models`：

- `download-models` 下载并校验固定模型，缓存到 `$CARGO_HOME/anr/models/dfn3-fe5eb64fa2e4/`，默认 Cargo home 为 `~/.cargo`。
- `copy-models` 将准备好的模型复制到 Cargo profile、examples 和测试输出目录中的 `models/dfn3.onnx`。
- **构建前**将 `ANR_MODEL_PATH` 指向本地模型可跳过下载；单独启用 `copy-models` 时必须设置。建议使用绝对路径保证构建可复现。缓存缺失且 `CARGO_NET_OFFLINE=true` 时直接报错。
- `cargo install` 可使用持久的构建时模型源路径。迁移到另一机器时，应在可执行文件旁提供模型及其许可/通知，不能依赖原机器缓存。

关闭默认 features 会同时去掉自动模型准备和默认 ORT 支持。模型准备见[库参考](../anr/README.md)，ORT 模型端口要求见 [ORT 参考](../anr-ort/README.md)。原生后端要求原模型精确匹配以下 SHA-256：

```text
fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9
```

其他导出即使张量形状相同，也不代表兼容原生后端。

### denoise 参数

```text
anr denoise INPUT... --output PATH [OPTIONS]
anr denoise INPUT... --output-dir DIR [OPTIONS]
```

| 参数 | 行为与默认值 |
| --- | --- |
| `INPUT...` | 一个或多个明确路径。`-` 表示 stdin，只能单输入并使用 `--output`。不递归遍历目录，不在程序内展开 glob。 |
| `-o, --output PATH` | 单输入目标；`-` 表示 stdout。与 `--output-dir` 互斥。 |
| `--output-dir DIR` | 必要时创建目录，串行生成 `stem.wav`，明确选择 raw 时生成 `stem.f32le`。同名目标报错，不自动改名。 |
| `--model PATH` | 可选覆盖路径，查找顺序见上文。`--model -` 指真实文件，不是 stdin。 |
| `--backend ort\|gl\|gles\|vk\|wgpu\|dx\|cuda\|tensorrt\|openvino\|opencl` | 默认 `ort`，native-only 构建也一样。须明确选择已编入的后端。 |
| `--provider cpu\|cuda\|tensorrt\|directml\|coreml\|xnnpack` | 仅适用于 ORT，省略时为 `cpu`。任何原生后端与显式 provider 同用均为参数错误。 |
| `--openvino-device DEVICE` | 仅适用于 OpenVINO：`CPU`（默认）、`GPU`、`GPU.n`（如 `GPU.0`）、`NPU`。不选 AUTO/HETERO，请求 GPU/NPU 失败不回退 CPU。 |
| `--strength VALUE` | 默认 `80`，接受有限 `0..=100`，不静默修正非法参数。零保留 DFN3 衰减分支，不是原音旁路。 |
| `--input-format auto\|f32le` | 默认 `auto` 探测带头音频；raw PCM 必须明确选 `f32le`。 |
| `--input-rate HZ` / `--input-channels N` | raw 必填正数，采样率为 `u32`、声道数为 `u16`。`auto` 不接受，不可覆盖音频头。 |
| `--output-format wav\|f32le` | 文件按 `.wav` / `.f32le` 扩展名识别，忽略大小写；其他扩展名需明确指定。目录输出默认 WAV；stdout 必须明确选 `f32le`。 |
| `--output-rate HZ` | 正 `u32`，默认输入采样率。 |
| `--wav-format pcm16\|pcm24\|float32` | 默认 `pcm16`；raw 输出显式设置此项报错。 |
| `--overwrite` | 只在处理成功后替换已有目标；仍禁止覆盖输入、所选模型和本批次已保存的结果。 |
| `-q, --quiet` / `-v, --verbose` | 互斥全局选项。quiet 保留错误；verbose 增加处理规格。诊断和完成统计写 stderr。 |
| `-h, --help` / `-V, --version` | 显示帮助 / 顶层版本并退出。 |

创建输出前先完成全局参数、目标冲突、运行库/provider 和首个模型实例的检查；输出目录创建后再次检查。输入、模型的路径和文件身份用于保护别名、符号链接、硬链接和大小写不敏感冲突，每个文件处理前及提交前复核。后续目标若成为已完成结果的别名，会报错而非覆盖。

每声道独立引擎，尽量复用，文件间重置。单文件解码、推理或输出失败会删除其临时输出并继续其他文件；全局初始化失败、新增声道引擎初始化失败、取消和 BrokenPipe 则停止批次。已成功文件保留，不整批回滚。文件已保存但无法核验其身份时，会明确报告并停止。

文件写到目标同目录的临时文件，推理、编码及 WAV finalize 全部成功后才提交；默认提交不覆盖并发创建的目标。即使使用 `--overwrite`，也不提前截断旧文件，但不保证断电持久性。`--output-dir` 创建所需父目录；使用 `--output` 时父目录必须已存在。

### 格式、时长与延迟

使用 Symphonia 0.5.5 的 `all` features 探测容器和解码，包括 WAV/PCM、FLAC、MP3、Ogg/Vorbis、AAC-LC/M4A。容器可识别不代表其中每种 codec/profile 都受支持，例如当前 codec 集合不含 Opus。选择第一个具有可用解码器的音轨；处理过程中切换音轨、采样率或声道布局会报错。

带头 stdin 不可 seek，需要 seek 的容器可能失败；CLI 不缓存整条流，也不暗中落盘。固定宽度 WAV 有可靠声明帧数时会校验提前 EOF；未知长度 WAV 标记仍可流式读取。压缩格式估算时长不作精确完整性预算，`inspect` 也不是整文件完整性校验。

输出为 WAV（PCM16、PCM24、float32）或 raw f32le，不提供 MP3/AAC 编码。WAV 写入采样和必要头部，不保留原标签、封面、章节或特殊声道布局元数据。整数采样四舍五入并限制范围；输入/输出出现非有限采样会报错。WAV 受 RIFF 32 位长度字段限制，不支持 RF64 输出。

管线保留声道数量和排列，不自动混音，使用有界流式缓冲，转换到 DFN3 的 **48 kHz / 480 采样**帧，再转到输出采样率；同采样率阶段绕过重采样。解码包之间积攒模型帧，只在 EOF 补最后一帧，随后裁掉补零。

输出长度按**实际解码帧数**而非头部估算计算：原采样率下等于输入帧数，换采样率时为 `round(输入帧数 × 输出采样率 / 输入采样率)`（半入）。这里一帧指同一时刻每声道各一个采样。重采样滤波延迟会补偿，并排出滤波尾部；**DFN3 自身算法延迟不补偿**，后端也没有模型流尾 flush 接口。等长不等于零延迟。管线架构见 [DESIGN.md](DESIGN.md)。

### 二进制安全的 raw 流

raw 为无头、交错排列、小端 IEEE 754 `f32`；N 声道每帧 `4 × N` 字节。EOF 时不完整采样或声道帧报错。stdin 和音频 stdout 不能是交互终端。stdout 只支持 raw f32le；WAV 需要 seek 回填头部。

Unix shell 示例（自动查找模型）：

```sh
anr denoise - --input-format f32le --input-rate 48000 --input-channels 1 --output - --output-format f32le < input.f32le > clean.f32le
```

Windows `cmd.exe` 示例（显式模型）：

```bat
anr denoise - --input-format f32le --input-rate 48000 --input-channels 1 --output - --output-format f32le --model "C:\models\dfn3.onnx" < input.f32le > clean.f32le
```

stderr 必须分离，音频流不要使用 `2>&1`。PowerShell 5.1 文本管道及 `Get-Content` / `Set-Content` 不保证二进制字节；PowerShell 7.4+ 支持原生命令字节管道和 stdout 重定向，见[官方说明](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_redirection?view=powershell-7.5)。

调用程序可直接传二进制句柄，避开 shell 转换；Python 只是可选示例：

```python
import subprocess

with open("input.f32le", "rb") as source, open("clean.f32le", "wb") as target:
    subprocess.run([
        "anr", "denoise", "-", "--input-format", "f32le",
        "--input-rate", "48000", "--input-channels", "1",
        "--output", "-", "--output-format", "f32le",
    ], stdin=source, stdout=target, check=True)
```

重定向或调用程序自行打开/截断 `clean.f32le`，该文件**不受** CLI 文件事务或 `--overwrite` 检查保护。stdout 已发送音频无法撤回，必须检查退出码和 stderr 再接受为完整流。

| 退出码 | 含义 |
| --- | --- |
| `0` | 成功、help/version，或下游提前关闭 stdout 导致的 `BrokenPipe`。BrokenPipe 成功不代表已输出整段录音。 |
| `1` | 运行失败，包括批次处理结束后有文件失败。 |
| `2` | 参数/用法错误。 |
| `130` | Ctrl+C 取消。 |

取消会退出处理循环并清理当前临时文件，已提交文件保留。stdin/stdout 阻塞时也无需先发送 EOF 才能响应。取消是操作之间的协作检查，不保证中断正在执行的厂商推理调用。

### 后端构建与部署

| 构建 feature | 运行选择 | 要求 |
| --- | --- | --- |
| `ort-download-binaries`（默认） | `--backend ort --provider cpu` | 构建时在可用平台下载预编译 CPU 运行库。 |
| `ort` | `--backend ort` | 自备运行库，通过 `ORT_LIB_LOCATION` 配置链接。 |
| `ort-load-dynamic` | `--backend ort` | 兼容 API 27 的共享库，`ORT_DYLIB_PATH` 指库文件；关闭默认 features 时不引入 ORT 下载器/TLS 依赖。 |
| `ort-cuda`、`ort-tensorrt`、`ort-directml`、`ort-coreml`、`ort-xnnpack` | `--provider cuda`、`tensorrt`、`directml`、`coreml`、`xnnpack` | 匹配的 ORT 运行库、平台、provider 库和驱动。DirectML 仅 Windows，CoreML 仅 macOS。 |
| `gl` | `--backend gl` | 硬件桌面 GL 4.3：Windows 或 Linux EGL/pbuffer。 |
| `gles` | `--backend gles` | 真实原生 ES 3.1+：Windows WGL 或 Linux 硬件 EGL/pbuffer，不接受桌面 GL 的 ES 兼容模式；无 Android/WebGL/macOS context 创建。 |
| `vk` | `--backend vk` | Windows/Linux 硬件 Vulkan 1.1，不经过 wgpu。 |
| `wgpu` | `--backend wgpu` | Windows/Linux/macOS 原生硬件 Vulkan/DX12/Metal adapter。 |
| `dx` | `--backend dx` | Windows 硬件 D3D12 与系统 D3DCompiler 47，不是 DirectML。 |
| `cuda` | `--backend cuda` | NVIDIA Driver API 12.0+、NVRTC 12/13（优先 13），驱动还须支持产生的 PTX；构建不需 CUDA SDK。 |
| `tensorrt` | `--backend tensorrt` | 构建需 TensorRT/CUDA SDK 与 C++17 编译器；运行需匹配库和 NVIDIA 驱动。 |
| `openvino` | `--backend openvino [--openvino-device CPU]` | 构建需 OpenVINO C++ SDK 与 C++17 编译器；运行需 Runtime、ONNX frontend、设备 plugin 及依赖。 |
| `opencl` | `--backend opencl` | OpenCL ICD loader 和硬件 GPU 厂商运行库/驱动，支持 OpenCL C 1.2+、在线编译及所需 FP32/设备能力；构建不需 SDK。 |

features 只编入支持，不安装 SDK 或驱动。ORT 允许 GPU provider 与逐算子 CPU 分配共存，显式 provider 注册/初始化失败仍报错。原生 GPU 后端不选择软件/CPU 回退。OpenVINO 默认明确选 CPU，请求 GPU/NPU 失败不回退，NPU 运算精度取决于设备。ORT CUDA/TensorRT/DirectML session 通过 `anr-ort` 在内存准备图，不改原模型，也不需 Python；见 [ORT 参考](../anr-ort/README.md)。GPU 内存加载目前要求模型目录可表示为 UTF-8，以解析外部权重。

已知 OpenVINO GPU 兼容性问题：原图编译曾在 `NonZero_1_count` / `count_nonzero` kernel 选择处失败。选择 GPU 不会绕过不支持的算子；NPU 支持同样取决于安装的 plugin 和设备。

```text
cargo build -p anr-cli --release
cargo build -p anr-cli --features ort-directml
cargo build -p anr-cli --no-default-features
cargo run -p anr-cli --release --no-default-features --features gles -- denoise input.wav --model /path/to/dfn3.onnx --backend gles --output clean.wav
```

无 feature 构建可检查音频、列出后端，但不能降噪。原生 features 不启用 ORT；用 `--no-default-features` 排除 ORT。native-only 构建必须明确传 `--backend`，参数默认值仍是 `ort`。

SDK 构建前先配置真实路径（PowerShell 示例）：

```powershell
$env:TENSORRT_ROOT = "C:\sdk\TensorRT"
$env:CUDA_ROOT = "C:\sdk\CUDA"
$env:OPENVINO_ROOT = "C:\sdk\openvino"
cargo build -p anr-cli --release --no-default-features --features cuda,tensorrt,openvino,opencl
```

TensorRT 可用 `CUDA_PATH` 代替 `CUDA_ROOT`，并以 `TENSORRT_LIB_DIR` / `CUDA_LIB_DIR` 覆盖链接库目录。OpenVINO 还接受 SDK/CMake 布局内的 `OpenVINO_DIR`，或官方 wheel 中含 `include/`、`libs/` 的 `openvino` 目录；生成的 CLI 不依赖 Python 运行。

自行部署时：

- CUDA 在 Windows 加载 `nvrtc64_130_0.dll` / `nvrtc64_120_0.dll`，Linux 加载 `libnvrtc.so.13` / `.12` / 无版本名库。匹配的 NVRTC/builtins 应在 `PATH` 或 `LD_LIBRARY_PATH`；仅有 Driver API 12.0 不保证兼容 NVRTC 13 PTX。
- TensorRT 需要构建 `OUT_DIR/anr_tensorrt_native.dll`（Linux 为 `libanr_tensorrt_native.so`）。`ANR_TENSORRT_SHIM` 指 shim 文件。仅 Windows 上 `ANR_TENSORRT_RUNTIME` 接受 TensorRT 库文件或 DLL 目录；Linux 忽略此变量，需通过平台 loader 路径（如 rpath 或 `LD_LIBRARY_PATH`）使库可加载。还需兼容 TensorRT/CUDA 库。目前没有持久 engine cache，构建 engine 会影响冷启动。
- OpenVINO 需要 `OUT_DIR` 中的 `anr_openvino_native.dll`（`libanr_openvino_native.so` / `.dylib`）。`ANR_OPENVINO_SHIM` 指 shim 文件，`ANR_OPENVINO_RUNTIME` 指 Runtime 库**文件**而非目录；保留 runtime/frontend/plugin 布局并配置平台库搜索路径。
- shim 放在可执行文件旁或设置覆盖路径。Cargo 安装不收集私有 shim 或厂商库，不能依赖另一构建机器记录的 SDK 路径。
- Windows TensorRT 按实际导入 DLL 名依次搜索 shim 目录、可执行文件目录、现存绝对 `PATH` 目录和构建时 SDK 目录，并去重。显式 `ANR_TENSORRT_RUNTIME`、`CUDA_ROOT` / `CUDA_PATH` 优先，配置错误直接失败，不改选其他安装。空/相对 `PATH` 不启用当前目录搜索，loader 不改写 `PATH`。
- ORT 共享链接除 `ORT_LIB_LOCATION` 外可能需要 `ORT_PREFER_DYNAMIC_LINK=1` 和平台库搜索路径。动态库加载失败在创建输出前返回 1。

硬件测试需主动启用：设置 `ANR_DFN3_MODEL`，使用 `cargo test -p anr-cli --release --no-default-features --features gl,gles,vk,wgpu,dx,cuda,opencl --test cli compute_ -- --include-ignored`，只选本机支持的 features；SDK 后端在配置对应 SDK 后再加入。测试覆盖流/文件行为，不代表感知音质或所有设备兼容性。性能记录单列于 [benchmark.md](../../benchmark.md)。

### 许可

项目代码采用 **MIT OR Apache-2.0**：[LICENSE-MIT](LICENSE-MIT)、[LICENSE-APACHE](LICENSE-APACHE)。所提供模型的来源和授权见[模型通知](../../.github/models/NOTICE.md)。依赖、运行库、模型和 SDK 保留各自条款；再分发及源码提供义务见 [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)。
