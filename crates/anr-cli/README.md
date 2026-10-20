# anr-cli

[English](#english) | [简体中文](#简体中文)

## English

`anr-cli` installs the `anr` command for denoising audio files, processing batches one file at a time, and reading or writing audio through stdin/stdout. It calls DFN3 / ONNX Runtime through `anr` and handles decoding, band-limited resampling, and multiple channels itself. You don't need FFmpeg or Python to run it.

Building from source requires **Rust 1.88 or later**. You must supply a model explicitly: none is packaged, downloaded automatically, or picked up from a `models/` directory in the checkout. It must have exactly these three inputs and three outputs, all `f32`:

| Direction | Name | Runtime shape |
| --- | --- | --- |
| Input | `input_frame` | `[480]` |
| Input | `states` | `[45304]` |
| Input | `atten_lim_db` | `[]` or `[1]` |
| Output | `enhanced_audio_frame` | `[480]` |
| Output | `new_states` | `[45304]` |
| Output | `lsnr` | `[]` or `[1]` |

A one-dimensional tensor may declare a symbolic length, but runtime lengths still have to match. Check the model's usage rights, sound quality, and algorithmic latency separately.

### Install and try it

Install from the source workspace root. The default build uses ORT's prebuilt libraries where available. **On Intel macOS, use the instructions below for supplying your own runtime instead.**

```text
cargo install --path crates/anr-cli --locked
anr --help
anr backends --json
```

The executable is `anr`, not `anr-cli`. Add the installation directory to `PATH`, or run the executable by its full path. `0.1.0` is the current project version, not a claim that it has been published on crates.io. Only after that version is actually published can you install it from the registry with `cargo install anr-cli --version 0.1.0 --locked`.

The default `ort` feature uses CPU ONNX Runtime. Building may download runtime libraries and requires network access, a target toolchain, and platform support:

- Windows: the MSVC C++ toolchain and Windows SDK.
- Linux: C/C++ linking tools, `pkg-config`, and OpenSSL development files, such as Ubuntu's `libssl-dev`.
- macOS: Xcode Command Line Tools.

The prebuilt distribution for `ort 2.0.0-rc.13` does not include Intel macOS. On that platform, build ONNX Runtime yourself or supply a compatible shared library. Installing the Rust target alone is not enough.

For example, once you have a compatible CPU `libonnxruntime.dylib` on Intel macOS, set its actual absolute path in the environment used to run `anr`, then install the dynamic-loading build:

```sh
export ORT_DYLIB_PATH="/absolute/path/to/libonnxruntime.dylib"
cargo install --path crates/anr-cli --locked --no-default-features --features ort-load-dynamic
anr backends --json
```

Replace that path with a real file and keep the setting for later runs. `backends` lists compiled support; only `denoise` loads and uses the runtime. See the [ONNX Runtime CPU build instructions](https://onnxruntime.ai/docs/build/inferencing.html) to build it yourself. CI pins [Microsoft's v1.28.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0) and does not assume that release provides Intel macOS binaries. CMake and Python are build tools in that case, not audio-processing dependencies of the installed CLI.

Denoise a file, process a batch, or inspect a recording:

```text
anr denoise voice.wav --output clean.wav --model /path/to/dfn3.onnx
anr denoise a.flac b.mp3 --output-dir cleaned --model /path/to/dfn3.onnx
anr inspect voice.wav
anr inspect voice.wav --json
```

Replace `/path/to/dfn3.onnx` with a model you're allowed to use. Quoted Windows paths work too. Relative paths are resolved from the command's working directory, not the executable or source directory. `inspect` does not load a model. `backends` reports what was compiled in; it does not prove a runtime or GPU is ready to use.

### Arguments and batches

```text
anr denoise INPUT... --output PATH --model MODEL [OPTIONS]
anr denoise INPUT... --output-dir DIR --model MODEL [OPTIONS]
anr inspect INPUT [--json]
anr backends [--json]
```

| Argument | Meaning |
| --- | --- |
| `INPUT...` | Explicit file paths. For a single input, `-` means stdin. The program does not scan directories recursively or expand globs itself. |
| `--output PATH` | Destination for one input; `-` means stdout. Cannot be combined with `--output-dir`. |
| `--output-dir DIR` | Processes files serially, using each input's stem with `.wav`, or `.f32le` when raw output is selected. Does not rename colliding outputs. |
| `--model PATH` | Required model path. |
| `--backend ort` / `--provider cpu` | Current backend and default execution provider. Other providers need the corresponding Cargo feature and runtime environment. |
| `--strength 80` | Defaults to 80. Accepts finite values in `0..=100`; rejects NaN, infinity, and out-of-range values rather than clamping them. |
| `--input-format auto\|f32le` | Defaults to detecting audio with headers. Raw input requires explicit `f32le`. |
| `--input-rate HZ` / `--input-channels N` | Required positive values for raw input. Cannot override the specifications in an audio header. |
| `--output-format wav\|f32le` | Inferred from `.wav` / `.f32le` for files; other extensions need an explicit format. Stdout requires explicit `f32le`. |
| `--output-rate HZ` | Optional positive sample rate. Defaults to the input rate. |
| `--wav-format pcm16\|pcm24\|float32` | Defaults to PCM16. Cannot be explicitly set for raw output. |
| `--overwrite` | Allows replacing an existing output after processing finishes. Never allows overwriting any input in the current run. |
| `--quiet` / `--verbose` | Mutually exclusive global options for stderr diagnostics. Do not change audio stdout. |

Before writing output, batch processing checks global arguments, the model and provider, and explicit destination conflicts. The model instance is reused between files, with state reset before each new file. If one file fails, its temporary output is removed and the remaining files continue. The final exit code is 1; files that already succeeded remain. There is no whole-batch rollback.

Each file is first written to a temporary file in the destination directory. It is moved to the final path only after inference, encoding, and WAV finalization finish. Existing files are not overwritten by default; even with `--overwrite`, the old file is not truncated before processing. This does not guarantee durability through power loss. Bytes already written to stdout cannot be taken back. Check the exit code and stderr before treating a stream as complete.

### Audio formats and duration

Input decoding and container detection use Symphonia 0.5.5 with its `all` features, including WAV/PCM, FLAC, MP3, Ogg/Vorbis, and AAC-LC/M4A support. **Recognizing a container does not mean every codec or profile inside it is supported.** Opus, for example, is not in the current codec set. Individual files still have to decode successfully; this list is not a claim that every encoding variant or platform has been tested.

For files with multiple tracks, the first decodable audio track is selected. `inspect` reports the selected track, sample rate, channels, and other details. Its JSON fields are `format`, `codec`, `track_id`, `sample_rate`, `channels`, `frames`, and `duration_seconds`; unknown frame counts or durations are `null`. Changes in sample rate or channel count during processing are rejected. Encoded, non-seekable stdin is also subject to demuxer limitations: inputs that need seeking may fail. The CLI does not work around this by buffering the whole input or silently writing it to disk.

Output is PCM16, PCM24, or float32 WAV, or raw f32le. There is no MP3/AAC encoder. WAV output contains the necessary audio headers and samples; it does not promise to preserve cover art, tags, chapters, or special channel-layout markers. Integer WAV samples are clamped to their representable range. Non-finite samples are errors.

Each channel has independent model state. Channel count and order are preserved; channels are not mixed automatically. Processing uses bounded buffers rather than loading the whole recording. Samples are collected across decoded packets into full model frames; only the final frame is padded at EOF. Audio is resampled to the model's 48 kHz and then to the output rate when needed; equal rates skip resampling. By default, the output frame count equals the number of frames actually decoded. With an explicit rate change, it is `round(input_frames × output_rate / input_rate)`. Here, a frame means one sample per channel at the same point in time.

Resampler filter delay is compensated and padding is trimmed. **DFN3's own algorithmic latency is not guessed or compensated**, and the backend has no end-of-stream flush operation. Equal input and output lengths do not mean model latency has been removed, nor do they guarantee better sound or real-time performance.

### Raw audio without corrupting the bytes

Raw audio is **little-endian IEEE 754 `f32`, with interleaved channels**. There is no header, sample rate, or channel metadata. Each frame contains N samples, or `4 × N` bytes. Leftover sample bytes or an incomplete channel frame at EOF cause an error.

```text
anr denoise - --input-format f32le --input-rate 48000 --input-channels 2 --output - --output-format f32le --model /path/to/dfn3.onnx
```

This shows the arguments only; connect binary files or a calling program to stdin/stdout. With `--output -`, stdout contains audio bytes only. Logs, statistics, and errors go to stderr—do not merge it into the audio stream. Stdout must be redirected so binary audio is not sent to a terminal. **WAV stdout is not supported** because writing the WAV header requires seeking.

In Windows `cmd.exe`, use native byte redirection, replacing the paths with real ones:

```bat
anr denoise - --input-format f32le --input-rate 48000 --input-channels 1 --output - --output-format f32le --model "C:\models\dfn3.onnx" < input.f32le > clean.f32le
```

Unix shells can use the same `< input.f32le > clean.f32le` redirection. Windows PowerShell 5.1 text pipelines and `Get-Content` / `Set-Content` do not preserve raw PCM bytes. PowerShell 7.4+ supports native-to-native byte pipelines and native stdout redirection; still keep stderr separate. See the [official PowerShell redirection documentation](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_redirection?view=powershell-7.5).

For a cross-platform caller, pass binary file handles directly without shell text conversion. This Python example is optional; Python is not a CLI runtime dependency:

```python
import subprocess

with open("input.f32le", "rb") as source, open("clean.f32le", "wb") as target:
    subprocess.run([
        "anr", "denoise", "-", "--input-format", "f32le",
        "--input-rate", "48000", "--input-channels", "1",
        "--output", "-", "--output-format", "f32le",
        "--model", "/path/to/dfn3.onnx",
    ], stdin=source, stdout=target, check=True)
```

| Exit code | Meaning |
| --- | --- |
| `0` | Success, help/version, or `BrokenPipe` when a downstream reader closes stdout early. |
| `1` | Runtime failure, or at least one failed file in a batch. |
| `2` | Invalid arguments. |
| `130` | Cancellation with Ctrl+C. |

Cancellation stops the processing loop and removes the current temporary file. A blocked standard stream does not require the caller to send EOF before cancellation can be handled.

### Build with execution providers

```text
cargo build -p anr-cli --release
cargo build -p anr-cli --no-default-features
cargo build -p anr-cli --features ort-directml
cargo build -p anr-cli --no-default-features --features ort-load-dynamic
```

Without default features, the CLI can still list compiled support and inspect audio, but cannot denoise with ORT. CLI features forward to `anr`: `ort`, `ort-load-dynamic`, `ort-cuda`, `ort-tensorrt`, `ort-directml`, `ort-coreml`, and `ort-xnnpack`. CUDA/TensorRT, DirectML, CoreML, and XNNPACK each need the appropriate platform, libraries, or drivers. Enabling a feature does not install them.

For example, build with DirectML on Windows and select `--provider directml`, or build with CoreML on macOS and select `--provider coreml`. The default remains `cpu`. Explicit provider registration or initialization failures are errors. CUDA, TensorRT, DirectML, and CoreML cannot silently fall back to CPU for the whole graph; unsupported operators may cause a failure. XNNPACK is itself a CPU provider and allows ordinary CPU kernels to handle operators it doesn't support. It does not promise to run every operator through XNNPACK. In `backends --json`, `compiled` means support was built in; `runtime_available: null` means runtime availability has not been checked, not that a model has run successfully.

For a dynamic-loading build, set `ORT_DYLIB_PATH` before starting the program, pointing to a shared library **file** that supports API 27, not a directory. To link against your own runtime instead, set `ORT_LIB_LOCATION` to the library directory and `ORT_PREFER_DYNAMIC_LINK=1` for shared linking; configure the system's shared-library search path at runtime too. Distribute the required runtime libraries and notices with the program. Copying the executable alone is not enough for every build configuration.

### What has been tested

CI is configured for four native runners: Windows x64 (`windows-2025`), Linux x64 (`ubuntu-24.04`), macOS arm64 (`macos-15`), and macOS x64 (`macos-15-intel`). Each uses stable and Rust 1.88 to build and run CLI tests, then installs the CLI and exercises audio processing with a control graph from an unrelated directory. Intel macOS builds CPU ONNX Runtime from source on the runner. A workflow file is not a test result: only a successful job run establishes that platform's result. Runner labels are documented by [GitHub](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

The small ONNX control graphs check processing, state, and stream formats. They are not DFN3 weights and do not establish real-model sound quality. Record real-model results, compressed format variants, optional providers, and minimum OS/runtime requirements separately. A Linux GNU build does not establish support for musl or arbitrarily old glibc. The tool does not provide real-time audio devices, virtual microphones, drivers, or a cloud service.

Earlier local checks recorded workspace tests and real DFN3 inference on Windows x64 / Rust 1.88 and WSL Ubuntu 24.04 x64 / Rust 1.99, including installed runs from unrelated working directories. Windows console cancellation and Linux SIGINT preserved existing outputs and removed the current temporary file. **macOS has not been run, and the GitHub Actions jobs have not been executed.** The local results do not stand in for those runs.

### License and redistribution

The project code is **MIT OR Apache-2.0**; choose either. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE). Including both full texts in each standalone source package is this project's distribution choice, not a Cargo requirement.

Dependencies and model weights keep their own licenses. Symphonia uses MPL-2.0; rubato, the Rust `ort` bindings, ONNX Runtime, and optional providers have separate terms and notices. See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for source availability and modification obligations. Check use and redistribution rights for model weights and exported ONNX files separately; they cannot be inferred from the DeepFilterNet source license.

## 简体中文

`anr-cli` 提供命令 `anr`，用于音频文件降噪、逐个文件的批处理，以及 stdin/stdout 音频管道。它通过 `anr` 调用 DFN3 / ONNX Runtime，自行完成解码、带限重采样和多声道处理。运行时不需要 FFmpeg 或 Python。

从源码构建需要 **Rust 1.88 或更新版本**。模型必须明确提供：不随包分发、不自动下载，也不会自行选用源码目录里的 `models/` 文件。模型只能有下表的三个输入和三个输出，类型都必须为 `f32`：

| 方向 | 名称 | 运行时的大小 |
| --- | --- | --- |
| 输入 | `input_frame` | `[480]` |
| 输入 | `states` | `[45304]` |
| 输入 | `atten_lim_db` | `[]` 或 `[1]` |
| 输出 | `enhanced_audio_frame` | `[480]` |
| 输出 | `new_states` | `[45304]` |
| 输出 | `lsnr` | `[]` 或 `[1]` |

一维长度可以用符号维度声明，但运行时长度不能改变。模型授权、音质和算法延迟需要另外核实。

### 安装与第一次运行

在源码工作区根目录安装。默认构建方式使用 ORT 提供的预编译库；**Intel macOS 请改用后面的自备运行库方式。**

```text
cargo install --path crates/anr-cli --locked
anr --help
anr backends --json
```

二进制名是 `anr`，不是 `anr-cli`。把安装目录加入 `PATH`，或用完整路径运行可执行文件。`0.1.0` 是当前项目版本，不代表它已经上传 crates.io。只有该版本实际发布后，才可用 `cargo install anr-cli --version 0.1.0 --locked` 从 registry 安装。

默认 feature `ort` 使用 CPU ONNX Runtime，构建可能下载运行库，需要联网、目标工具链和平台支持：

- Windows：MSVC C++ 工具链和 Windows SDK。
- Linux：C/C++ 链接工具链、`pkg-config` 和 OpenSSL 开发文件，例如 Ubuntu 的 `libssl-dev`。
- macOS：Xcode Command Line Tools。

当前 `ort 2.0.0-rc.13` 的预编译分发不包含 Intel macOS。该平台需要自己构建 ONNX Runtime，或使用兼容的共享库；只安装 Rust target 不够。

例如，在 Intel macOS 上准备好兼容的 CPU `libonnxruntime.dylib` 后，在运行 `anr` 的环境里设置实际绝对路径，再安装动态加载版本：

```sh
export ORT_DYLIB_PATH="/absolute/path/to/libonnxruntime.dylib"
cargo install --path crates/anr-cli --locked --no-default-features --features ort-load-dynamic
anr backends --json
```

把路径换成真实文件，并在以后运行时保留设置。`backends` 只列出编译时加入的支持，实际执行 `denoise` 才会加载并使用运行库。可按 [ONNX Runtime CPU 源码构建说明](https://onnxruntime.ai/docs/build/inferencing.html) 准备运行库；CI 固定构建 [微软 v1.28.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0)，不假设该 release 提供 Intel macOS 预编译包。此时 CMake、Python 等只用于源码构建，不是安装后 CLI 处理音频的依赖。

处理单文件、批处理和信息查询：

```text
anr denoise voice.wav --output clean.wav --model /path/to/dfn3.onnx
anr denoise a.flac b.mp3 --output-dir cleaned --model /path/to/dfn3.onnx
anr inspect voice.wav
anr inspect voice.wav --json
```

把 `/path/to/dfn3.onnx` 换成自己有权使用的实际模型路径，Windows 同样可用带引号的本机路径。相对路径均相对于启动命令时的工作目录，不相对于可执行文件或源码目录。`inspect` 不加载模型；`backends` 只说明编译时加入了哪些支持，不证明运行库或 GPU 已能使用。

### 参数与批处理

```text
anr denoise INPUT... --output PATH --model MODEL [OPTIONS]
anr denoise INPUT... --output-dir DIR --model MODEL [OPTIONS]
anr inspect INPUT [--json]
anr backends [--json]
```

| 参数 | 含义 |
| --- | --- |
| `INPUT...` | 明确列出的文件路径；单输入时可用 `-` 表示 stdin。程序不递归扫描目录，也不自行展开 glob。 |
| `--output PATH` | 单输入的目标文件；`-` 表示 stdout。与 `--output-dir` 互斥。 |
| `--output-dir DIR` | 逐个处理文件，按输入文件名去掉扩展名后生成 `.wav`，明确选择 raw 时生成 `.f32le`。同名结果不自动改名。 |
| `--model PATH` | 必填的模型路径。 |
| `--backend ort` / `--provider cpu` | 当前后端及默认执行 provider；其他 provider 需对应的 Cargo feature 和运行环境。 |
| `--strength 80` | 默认 80；必须是 `0..=100` 的有限数值，拒绝 NaN、无穷和越界值，不静默修正。 |
| `--input-format auto\|f32le` | 默认探测带头音频；raw 必须明确选 `f32le`。 |
| `--input-rate HZ` / `--input-channels N` | raw 输入必填的正数；带头音频不能用它们覆盖真实规格。 |
| `--output-format wav\|f32le` | 文件可由 `.wav` / `.f32le` 扩展名识别；其他扩展名须明确指定。stdout 必须明确选 `f32le`。 |
| `--output-rate HZ` | 可选正采样率；默认保留输入采样率。 |
| `--wav-format pcm16\|pcm24\|float32` | 默认 PCM16；raw 输出不能显式指定此参数。 |
| `--overwrite` | 允许处理完成后替换已有目标，仍禁止覆盖本次任意输入。 |
| `--quiet` / `--verbose` | 互斥的全局选项，控制 stderr 诊断，不改变音频 stdout。 |

批处理在开始输出前检查全局参数、模型、provider 和明确的目标冲突。文件间复用模型实例，但每个新文件开始时会重置状态。单个文件失败会清理该文件的临时输出，继续其余文件，最终退出码为 1；之前成功的文件保留，不会撤销整批结果。

文件先写到目标同目录的临时文件，推理、编码和 WAV 文件收尾全部完成后才移到最终路径。默认不覆盖；即使使用 `--overwrite`，也不会在开始处理前截断旧文件。这不保证断电后数据仍然完整。stdout 已发出的字节无法撤回；失败时必须检查退出码和 stderr，不能把部分音频流当完整结果。

### 音频格式和时长

带头输入使用 Symphonia 0.5.5 的解码器和容器探测，启用其 `all` features，包括 WAV/PCM、FLAC、MP3、Ogg/Vorbis、AAC-LC/M4A 等支持。**容器可识别，不代表里面所有 codec/profile 都能解码。** 例如 Opus 不在当前 codec 集合内。具体文件仍以解码结果为准；这里没有声称所有编码变体或平台都已实测通过。

多音轨默认选择第一个可解码音轨；`inspect` 显示所选音轨、采样率、声道等信息。JSON 字段为 `format`、`codec`、`track_id`、`sample_rate`、`channels`、`frames`、`duration_seconds`；未知帧数或时长为 `null`。处理过程中改变采样率或声道数会报错。不可 seek 的编码 stdin 还受 demuxer 限制，需要 seek 的输入可能失败；CLI 不会因此缓存整条输入或暗中写入磁盘。

输出是 PCM16、PCM24 或 float32 WAV，或 raw f32le；不提供 MP3/AAC 编码。WAV 只写必要音频头和采样，不承诺保留封面、标签、章节或特殊声道布局标记。整数 WAV 会限制到可表示范围；非有限采样会报错。

每声道使用独立模型状态，保留声道数量与排列，不自动混音。处理时使用有界缓冲，不把整段录音读入内存；跨解码包积攒完整模型帧，只在 EOF 补最后一帧。必要时先重采样到模型的 48 kHz，再转到输出采样率；采样率相同时跳过重采样。默认输出帧数等于实际解码输入帧数；明确换采样率时为 `round(输入帧数 × 输出采样率 / 输入采样率)`。这里一帧指每声道同一时刻的一个采样。

重采样器的滤波延迟会补偿，补零部分会裁剪，但 **DFN3 模型本身的算法延迟不会猜测或补偿**，后端也没有在音频结束后取出模型剩余结果的操作。输出等长不等于模型延迟已消除，也不保证音质提升或实时性能。

### 不损坏音频字节的 raw 管道

raw 格式是 **小端 IEEE 754 `f32`，声道交错**，没有文件头、采样率或声道信息。每帧有 N 个采样，共 `4 × N` 字节；EOF 仍有残余采样字节或不完整声道帧时会报错。

```text
anr denoise - --input-format f32le --input-rate 48000 --input-channels 2 --output - --output-format f32le --model /path/to/dfn3.onnx
```

上面只展示参数，必须通过文件或调用程序接入二进制 stdin/stdout。`--output -` 的 stdout 只有音频字节；日志、统计和错误均写 stderr，不要把它合并进音频流。stdout 必须重定向，以免向终端发送二进制音频。WAV 需要 seek 回填头部，**不支持 WAV stdout**。

Windows `cmd.exe` 中可以用原生字节重定向，将路径替换为实际值：

```bat
anr denoise - --input-format f32le --input-rate 48000 --input-channels 1 --output - --output-format f32le --model "C:\models\dfn3.onnx" < input.f32le > clean.f32le
```

Unix shell 可用相同的 `< input.f32le > clean.f32le` 重定向。Windows PowerShell 5.1 的文本管道、`Get-Content` / `Set-Content` 不能原样传递 PCM 字节。PowerShell 7.4+ 支持 native-to-native 字节管道和 native stdout 重定向，仍不要合并 stderr；见 [PowerShell 官方重定向说明](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_redirection?view=powershell-7.5)。

跨平台调用可直接传入二进制文件句柄，不经过 shell 文本转换。下面的 Python 只是可选调用示例，不是 CLI 运行依赖：

```python
import subprocess

with open("input.f32le", "rb") as source, open("clean.f32le", "wb") as target:
    subprocess.run([
        "anr", "denoise", "-", "--input-format", "f32le",
        "--input-rate", "48000", "--input-channels", "1",
        "--output", "-", "--output-format", "f32le",
        "--model", "/path/to/dfn3.onnx",
    ], stdin=source, stdout=target, check=True)
```

| 退出码 | 含义 |
| --- | --- |
| `0` | 成功、help/version，或下游提前关闭 stdout 的 `BrokenPipe`。 |
| `1` | 运行失败，或批处理中至少一个文件失败。 |
| `2` | 参数错误。 |
| `130` | Ctrl+C 主动取消。 |

取消时退出处理循环并清理当前临时文件；即使标准流阻塞，也不需要调用者先发送 EOF 才能响应取消。

### 编入不同执行 provider

```text
cargo build -p anr-cli --release
cargo build -p anr-cli --no-default-features
cargo build -p anr-cli --features ort-directml
cargo build -p anr-cli --no-default-features --features ort-load-dynamic
```

不启用默认 feature 时，仍可列出编译时加入的支持、检查音频，但不能通过 ORT 降噪。CLI 的同名 feature 转发到 `anr`：`ort`、`ort-load-dynamic`、`ort-cuda`、`ort-tensorrt`、`ort-directml`、`ort-coreml`、`ort-xnnpack`。CUDA/TensorRT、DirectML、CoreML 和 XNNPACK 分别需要相应平台、运行库或驱动；启用 feature 不会安装这些依赖。

例如，在 Windows 编入 DirectML 后选择 `--provider directml`，在 macOS 编入 CoreML 后选择 `--provider coreml`。默认仍为 `cpu`。明确指定的 provider 注册或初始化失败会报错；CUDA、TensorRT、DirectML、CoreML 不会在整张图无法执行时静默改用 CPU，遇到不支持的算子可能直接失败。XNNPACK 本身是 CPU provider，允许普通 CPU kernel 处理它不支持的算子，不代表全部算子都由 XNNPACK 运行。`backends --json` 中，`compiled` 只表示编译时加入了支持；`runtime_available: null` 表示尚未检查运行时可用性，不代表模型成功运行。

动态加载构建需在启动前设置 `ORT_DYLIB_PATH`，指向兼容 API 27 的共享库**文件**，不是目录。若要链接自备运行库，可用 `ORT_LIB_LOCATION` 指向库目录，并以 `ORT_PREFER_DYNAMIC_LINK=1` 选择共享库；运行时还需配置系统动态库搜索路径。分发时随程序提供必要的运行库和适用通知，不能认为所有构建方式都只需复制可执行文件。

### 哪些已经测过

CI 配置了四种原生 runner：Windows x64（`windows-2025`）、Linux x64（`ubuntu-24.04`）、macOS arm64（`macos-15`）、macOS x64（`macos-15-intel`）。每种分别用 stable 和 Rust 1.88 构建并运行 CLI 测试，再安装 CLI，从无关目录用控制图运行音频处理；Intel macOS 在 runner 内从源码构建 CPU ONNX Runtime。配置文件不是运行结果，只有对应 job 实际通过才能记录该平台通过。runner 标签来源见 [GitHub 官方文档](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)。

自制 ONNX 控制图只验证处理、状态和音频流格式，不是 DFN3 权重，也不证明真实模型音质。真实模型、压缩格式变体、可选 provider 及最低 OS/运行库要求，需要分别记录实际结果。Linux GNU 构建不代表支持 musl 或任意旧 glibc。本工具不提供实时音频设备、虚拟麦克风、驱动或云服务。

此前的本地检查记录了 Windows x64 / Rust 1.88 与 WSL Ubuntu 24.04 x64 / Rust 1.99 上的工作区测试和真实 DFN3 推理，也验证了安装后从无关工作目录运行。Windows 控制台取消和 Linux SIGINT 均保留原目标、清理当前临时文件。**macOS 尚未实跑，GitHub Actions jobs 也尚未执行。** 本地结果不能代替这些运行。

### 许可与再分发

项目代码采用 **MIT OR Apache-2.0**，任选其一；见 [LICENSE-MIT](LICENSE-MIT) 和 [LICENSE-APACHE](LICENSE-APACHE)。在独立源码包内附带两份完整文本是本项目的分发选择，不是 Cargo 的强制要求。

依赖和权重保留各自的许可。Symphonia 为 MPL-2.0；rubato、Rust `ort` 绑定、ONNX Runtime 及可选 provider 另有许可和通知。源码获取与修改义务见 [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)。模型权重和导出 ONNX 的使用与再分发授权需要单独确认，不能由 DeepFilterNet 源码许可推断。
