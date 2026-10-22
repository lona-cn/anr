# anr-cli

[English](#english) | [简体中文](#简体中文)

## English

`anr-cli` installs the `anr` command for denoising audio files, serial batches, and stdin/stdout audio. It calls DFN3 through `anr`: ONNX Runtime by default, or independent OpenGL, OpenGL ES, Vulkan, wgpu, Direct3D 12 Compute, CUDA, TensorRT, OpenVINO, and OpenCL backends without ORT. The CLI handles decoding, band-limited resampling, and multiple channels. Runtime use requires neither FFmpeg nor Python.

Source builds require the **latest stable Rust toolchain**. Default features download and prepare the pinned DFN3 model at build time, including with `cargo install`; application ZIPs bundle that model and CPU runtime. `--model` is optional: an explicit path wins, otherwise the CLI checks `models/dfn3.onnx` beside the executable and then the model source recorded at build time. No checkout-relative path is assumed. The ORT backend requires exactly these three inputs and three outputs, all `f32`:

| Direction | Name | Runtime shape |
| --- | --- | --- |
| Input | `input_frame` | `[480]` |
| Input | `states` | `[45304]` |
| Input | `atten_lim_db` | `[]` or `[1]` |
| Output | `enhanced_audio_frame` | `[480]` |
| Output | `new_states` | `[45304]` |
| Output | `lsnr` | `[]` or `[1]` |

A one-dimensional tensor may declare a symbolic length, but runtime lengths still have to match. These are ORT port checks, not the compute backends' compatibility test: compute requires the exact SHA-256 listed below. Check model rights, sound quality, and algorithmic latency separately.

### Install and try it

#### Use an application ZIP

The release workflow produces `anr-<version-or-commit>-<target>.zip` for Windows x64, Linux x64, macOS Apple Silicon, and macOS Intel after its platform checks pass. Each ZIP contains the executable, CPU ONNX Runtime **1.28.0**, `models/dfn3.onnx`, and applicable licenses. Processing audio needs no Rust, Python, or FFmpeg. The [model release](https://github.com/lona-cn/anr/releases/tag/model-dfn3-fe5eb64fa2e4) is a separate asset, not a CLI installation.

Extract everything, enter the extracted directory, and run:

```sh
./anr denoise input.wav --output clean.wav
```

On Windows use `.\anr.exe`. The bundled model is found automatically, regardless of the working directory. Keep the native shared libraries beside the executable. No `ORT_DYLIB_PATH` setting is needed for the bundled runtime; remove an old override if you want to use the bundled copy.

Windows may need the [Microsoft Visual C++ x64 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist). The Linux package targets glibc and is built on Ubuntu 24.04, not musl or arbitrary older glibc. macOS packages are ad-hoc signed, not notarized; follow the system's approval prompts for downloaded applications. GPU providers are not bundled. Check attached `SHA256SUMS` before use.

#### Build from source

Install from the source workspace root. The default build uses ORT's prebuilt libraries where available. **On Intel macOS, use the instructions below for supplying your own runtime instead.**

```text
cargo install --path crates/anr-cli --locked
anr --help
anr backends --json
```

The executable is `anr`, not `anr-cli`. Add the installation directory to `PATH`, or run the executable by its full path. To install version `0.1.1` from crates.io, use `cargo install anr-cli --version 0.1.1 --locked`.

The default features are `ort-download-binaries`, `download-models`, and `copy-models`: they enable the ORT backend, download the prebuilt CPU runtime and pinned model, and copy the model beside build outputs. The first build requires network access, a target toolchain, and platform support:

- Windows: the MSVC C++ toolchain and Windows SDK.
- Linux: C/C++ linking tools, `pkg-config`, and OpenSSL development files, such as Ubuntu's `libssl-dev`.
- macOS: Xcode Command Line Tools.

The prebuilt distribution for `ort 2.0.0-rc.15` uses ONNX Runtime 1.31.0 and does not include Intel macOS. On that platform, build ONNX Runtime yourself or supply an API-27-compatible shared library, including 1.28.0. Installing the Rust target alone is not enough.

For example, once you have a compatible CPU `libonnxruntime.dylib` on Intel macOS, set its actual absolute path in the environment used to run `anr`, then install the dynamic-loading build:

```sh
export ORT_DYLIB_PATH="/absolute/path/to/libonnxruntime.dylib"
cargo install --path crates/anr-cli --locked --no-default-features --features ort-load-dynamic,download-models,copy-models
anr backends --json
```

Replace that path with a real file and keep the setting for later runs. `backends` lists compiled support; only `denoise` loads and uses the runtime. See the [ONNX Runtime CPU build instructions](https://onnxruntime.ai/docs/build/inferencing.html) to build it yourself. CI pins [Microsoft's v1.28.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0) and does not assume that release provides Intel macOS binaries. CMake and Python are build tools in that case, not audio-processing dependencies of the installed CLI.

Denoise a file, process a batch, or inspect a recording:

```text
anr denoise voice.wav --output clean.wav
anr denoise a.flac b.mp3 --output-dir cleaned
anr inspect voice.wav
anr inspect voice.wav --json
```

Use `--model /path/to/dfn3.onnx` to select a different model you're allowed to use. Quoted Windows paths work too. Explicit relative paths are resolved from the command's working directory; automatic model discovery is relative to the executable, then the build-time source. `inspect` does not load a model. `backends` reports what was compiled in; it does not prove a runtime or GPU is ready to use.

`download-models` caches the verified model under `$CARGO_HOME/anr/models/dfn3-fe5eb64fa2e4/` (`~/.cargo` by default). `copy-models` independently copies it to `models/dfn3.onnx` in the Cargo profile, examples, and tests directories. Set `ANR_MODEL_PATH` to an absolute local file path **before building** to skip model downloading, or to use `copy-models` alone. A download-only build and `cargo install` can use the persistent source path without a bundled copy. For distribution to another machine, include the model and its applicable licenses/notices; the original build cache is not portable. No model is downloaded at runtime.

Disable default features to omit automatic model preparation and its downloader dependencies. Then pass `--model`, provide a bundled model, or opt back into the individual model features. A cold cache with `CARGO_NET_OFFLINE=true` fails rather than downloading; use `ANR_MODEL_PATH` for offline builds. See the [model preparation guide](../anr/README.md#model-download-and-copying).

### Arguments and batches

```text
anr denoise INPUT... --output PATH [--model MODEL] [OPTIONS]
anr denoise INPUT... --output-dir DIR [--model MODEL] [OPTIONS]
anr inspect INPUT [--json]
anr backends [--json]
```

| Argument | Meaning |
| --- | --- |
| `INPUT...` | Explicit file paths. For a single input, `-` means stdin. The program does not scan directories recursively or expand globs itself. |
| `--output PATH` | Destination for one input; `-` means stdout. Cannot be combined with `--output-dir`. |
| `--output-dir DIR` | Processes files serially, using each input's stem with `.wav`, or `.f32le` when raw output is selected. Does not rename colliding outputs. |
| `--model PATH` | Optional model override. Otherwise uses the bundled or build-time model; an explicit invalid path is an error, not a fallback. |
| `--backend ort\|gl\|gles\|vk\|wgpu\|dx\|cuda\|tensorrt\|openvino\|opencl` | Defaults to `ort`. Each native backend requires its matching opt-in Cargo feature and the exact supported model below. |
| `--provider cpu` | ORT only; omitted means `cpu` for ORT. Other providers require matching features and runtime support. Any explicit `--provider` with a non-ORT backend is an argument error. |
| `--openvino-device CPU\|GPU\|GPU.n\|NPU` | OpenVINO only; omitted means explicit `CPU`. `GPU.n` selects a GPU index, such as `GPU.0`. GPU/NPU failures do not fall back to CPU. Invalid use with another backend is an argument error. |
| `--strength 80` | Defaults to 80. Accepts finite values in `0..=100`; rejects NaN, infinity, and out-of-range values rather than clamping them. |
| `--input-format auto\|f32le` | Defaults to detecting audio with headers. Raw input requires explicit `f32le`. |
| `--input-rate HZ` / `--input-channels N` | Required positive values for raw input. Cannot override the specifications in an audio header. |
| `--output-format wav\|f32le` | Inferred from `.wav` / `.f32le` for files; other extensions need an explicit format. Stdout requires explicit `f32le`. |
| `--output-rate HZ` | Optional positive sample rate. Defaults to the input rate. |
| `--wav-format pcm16\|pcm24\|float32` | Defaults to PCM16. Cannot be explicitly set for raw output. |
| `--overwrite` | Allows replacing an existing output after processing finishes. Never permits replacing an input, model, or a result already completed in this batch. |
| `--quiet` / `--verbose` | Mutually exclusive global options for stderr diagnostics. Do not change audio stdout. |

Before writing output, batch processing checks global arguments, the model/backend (and ORT provider when applicable), and resolvable destination conflicts. Actual file identities are checked again before processing and before committing, including case-insensitive aliases. If a previously nonexistent alias is discovered after an earlier batch result was saved, the later file fails instead of replacing that result. Each channel has an independent reducer, reused with state reset between files. Failed files remove their temporary output and processing continues; the final exit code is 1 and successful files remain. If a saved file's identity cannot be verified, the CLI reports that it was saved and stops further processing. There is no whole-batch rollback.

Each file is first written to a temporary file in the destination directory. It is moved to the final path only after inference, encoding, and WAV finalization finish. Existing files are not overwritten by default; even with `--overwrite`, the old file is not truncated before processing. This does not guarantee durability through power loss. Bytes already written to stdout cannot be taken back. Check the exit code and stderr before treating a stream as complete.

### Audio formats and duration

Input decoding and container detection use Symphonia 0.5.5 with its `all` features, including WAV/PCM, FLAC, MP3, Ogg/Vorbis, and AAC-LC/M4A support. **Recognizing a container does not mean every codec or profile inside it is supported.** Opus, for example, is not in the current codec set. Individual files still have to decode successfully; this list is not a claim that every encoding variant or platform has been tested.

For files with multiple tracks, the first decodable audio track is selected. `inspect` reports the selected track, sample rate, channels, and other details. Its JSON fields are `format`, `codec`, `track_id`, `sample_rate`, `channels`, `frames`, and `duration_seconds`; unknown frame counts or durations are `null`. Changes in sample rate or channel count during processing are rejected. Encoded, non-seekable stdin is also subject to demuxer limitations: inputs that need seeking may fail. The CLI does not work around this by buffering the whole input or silently writing it to disk.

Fixed-width WAV audio with a reliable declared frame count is checked against the frames actually decoded. Premature EOF is an error and cannot commit a short replacement. Unknown-length WAV markers remain streamable; estimated durations from compressed formats are not treated as exact integrity budgets. This is not a guarantee that every damaged container or codec can be detected.

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

### Build with backends and execution providers

```text
cargo build -p anr-cli --release
cargo build -p anr-cli --no-default-features
cargo build -p anr-cli --features ort-directml
cargo build -p anr-cli --no-default-features --features ort-load-dynamic
cargo build -p anr-cli --no-default-features --features gles
cargo build -p anr-cli --no-default-features --features gl,gles,vk,wgpu,dx
```

With no features, the CLI can list compiled support and inspect audio but cannot denoise. `ort` alone enables the backend and links a runtime supplied through `ORT_LIB_LOCATION`; the default `ort-download-binaries` adds CPU runtime downloads. `--no-default-features --features ort-load-dynamic` instead loads your shared library without the downloader or TLS/OpenSSL build dependencies. The `ort-cuda`, `ort-tensorrt`, `ort-directml`, `ort-coreml`, and `ort-xnnpack` flags forward ORT providers to `anr` and need the appropriate platform, libraries, or drivers; a feature does not install them. They are distinct from the native `cuda` and `tensorrt` features.

For example, build with DirectML on Windows and select `--provider directml`, or build with CoreML on macOS and select `--provider coreml`. The default remains `cpu`. The project normally allows ORT to place individual operators on CPU alongside CUDA, TensorRT, DirectML, or CoreML when unsupported by that provider or preferred by ORT's placement rules. This is the default policy, not a temporary option; selecting a GPU provider does not promise GPU-only execution. Explicit provider registration or initialization failures remain errors and do not silently replace an unavailable provider with CPU. XNNPACK is itself a CPU provider and also allows ordinary CPU kernels. In `backends --json`, `compiled` means support was built in; `runtime_available: null` means runtime availability has not been checked, not that a model has run successfully.

For ORT's CUDA/TensorRT providers, the CLI automatically prepares an in-memory model before creating each session, using the same `anr-ort` entry point as the benchmark. No extra flag, converted model file, or Python is required. The original ONNX and weights remain unchanged; attenuation stays adjustable. CPU/DirectML and native backends keep their existing loading paths.


For a dynamic-loading build, set `ORT_DYLIB_PATH` before starting the program, pointing to an API-27-compatible shared library **file**, not a directory. Loading errors return exit code 1 before output is created. The CLI prepares one environment for the batch; library users own their environment explicitly. For shared linking, set `ORT_LIB_LOCATION` and `ORT_PREFER_DYNAMIC_LINK=1`, and configure the system's library search path too. Distribute the required libraries and notices; copying the executable alone is not enough for every configuration.

The optional `gl`, `gles`, `vk`, `wgpu`, `dx`, `cuda`, `tensorrt`, `openvino`, and `opencl` features enable native backends without enabling ONNX Runtime. Use `--no-default-features` to omit ORT, and enable any subset supported by your environment. The default `--backend` remains `ort` even if ORT was not compiled; a native-only build must explicitly select its backend. It cannot be combined with `--provider`, including `--provider cpu`.

For a GLES-only source run, use Cargo so the command runs the binary just built, not an older `anr` on `PATH`:

```text
cargo run -p anr-cli --release --no-default-features --features gles -- denoise input.wav --model /path/to/dfn3.onnx --backend gles --output clean.wav
```

The GL/GLES/Vulkan/wgpu/D3D12 backends use the original ONNX as a parameter container, not a generic executable graph. The fixed model must have SHA-256 `fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9`; other exports are rejected even with identical port shapes. They share the fixed execution plan and WGSL operators through `anr-dfn3`, execute FFT, neural network, and synthesis on GPU, and preserve 48 kHz / 480-sample framing, CLI resampling, independent channel state, strength, reset, and exact output length. Strength zero is the original attenuation branch, not an audio bypass.

`gles` requires a real native ES 3.1+ context on Windows/WGL or Linux hardware EGL/pbuffer. It shares `anr-gl`'s executor, generates GLSL ES 310, and maps the output frame after a fence instead of calling desktop-only `glGetBufferSubData`. Desktop-GL ES compatibility is not accepted. This backend does not provide Android, WebGL, or macOS context creation.

`gl` requires Windows desktop GL 4.3 or Linux hardware EGL with desktop GL 4.3/pbuffer; `vk` requires Windows/Linux hardware Vulkan 1.1; `wgpu` uses a native hardware Vulkan/DX12/Metal adapter on Windows/Linux/macOS; `dx` uses Windows hardware D3D12 and system D3DCompiler 47. Native VK/DX do not go through wgpu, and DX is not DirectML. No backend silently selects CPU/software rendering. Unsupported hardware or initialization failures are errors; macOS GPU execution has not been verified.

Do not pass `--provider` with native backends, including `--provider cpu`. `backends --json` lists all ten names without loading runtimes or creating a GPU device. Native entries have no ORT providers and leave `runtime_available` as `null`: this means availability was not probed, not that libraries or devices are ready. Verbose diagnostics identify the selected backend; only ORT emits provider diagnostics.

Real-model CLI tests are opt-in. Set `ANR_DFN3_MODEL` and run `cargo test -p anr-cli --no-default-features --features gl,gles,vk,wgpu,dx --test cli compute_ -- --include-ignored`, enabling only features supported by your machine. These tests exercise raw/WAV, independent stereo state, partial-frame length, batch reset, and strength. Every enabled compute backend needs real hardware; an ignored test is not a passing hardware result.

### Native CUDA, TensorRT, OpenVINO, and OpenCL deployment

These four opt-in backends use the same fixed model SHA-256 above and preserve CLI resampling, independent channel state, strength, reset, partial-frame handling, and output length. CUDA and OpenCL run the complete fixed FP32 DSP/FFT/neural plan on GPU. CUDA uses Driver/NVRTC APIs and replay graphs; OpenCL requires a hardware GPU ICD with OpenCL 1.2+, not a CPU implementation. TensorRT builds two real standard-layer neural engines plus CUDA DSP/FFT graphs, with strongly typed FP32 and TF32 disabled; it uses neither ORT, the ONNX parser, nor an inference plugin. There is no persistent TensorRT engine cache: initialization took 17.342 seconds in the recorded run, so do not assume fast cold startup.

OpenVINO imports the original ONNX through its native Runtime, makes the recurrent state stateful, and creates a separate InferRequest for each stream. The default is explicit CPU, never AUTO/HETERO. `--openvino-device GPU`, `GPU.0`, or `NPU` requests that device only; missing devices or unsupported operators are errors, not CPU fallbacks. NPU arithmetic is device-dependent and is not promised to be FP32 merely because the model is FP32.

CUDA and OpenCL need no build-time vendor SDK for their Rust crates, but still need compatible runtime libraries and drivers. TensorRT and OpenVINO require an existing SDK and a C++17 compiler at build time; their `build.rs` compiles a private shared shim, loaded by Rust through `libloading`. Nothing installs an SDK system-wide or downloads one automatically.

```text
cargo build -p anr-cli --release --no-default-features --features cuda,opencl
cargo run -p anr-cli --release --no-default-features --features cuda -- denoise input.wav --model /path/to/dfn3.onnx --backend cuda --output clean.wav
```

For SDK-enabled builds, configure actual SDK paths in the build environment first. For example in PowerShell (replace these paths):

```powershell
$env:TENSORRT_ROOT = "C:\sdk\TensorRT"
$env:CUDA_ROOT = "C:\sdk\CUDA"
$env:OPENVINO_ROOT = "C:\sdk\openvino"
cargo build -p anr-cli --release --no-default-features --features cuda,tensorrt,openvino,opencl
```

TensorRT needs headers and link libraries under `TENSORRT_ROOT` and `CUDA_ROOT` (or `CUDA_PATH`); `TENSORRT_LIB_DIR` and `CUDA_LIB_DIR` can override the link-library directories. OpenVINO accepts `OPENVINO_ROOT` or `OpenVINO_DIR` pointing into its SDK/CMake layout. The official OpenVINO wheel's `openvino` directory with `include/` and `libs/` is also a supported SDK source; Python is not a product runtime dependency.

After building, use `--backend tensorrt`, `--backend opencl`, or `--backend openvino --openvino-device CPU`. In contrast, ORT CUDA uses the `ort-cuda` feature and `--backend ort --provider cuda`; ORT TensorRT uses `ort-tensorrt` and `--backend ort --provider tensorrt`. Native backends never inherit ORT's operator-level CPU placement policy.

For custom application distribution, include the model and notices plus the following dependencies; do not rely on absolute paths embedded from the build machine:

| Backend | Runtime deployment requirements |
| --- | --- |
| CUDA | NVIDIA Driver API 12.0+ and NVRTC major 12 or 13 (13 preferred), with a driver capable of loading the PTX produced by the selected NVRTC. Driver API 12.0 support alone does not guarantee compatibility with NVRTC 13 PTX. Windows loads `nvrtc64_130_0.dll` / `nvrtc64_120_0.dll`; Linux loads `libnvrtc.so.13` / `.12` / unversioned. Put matching NVRTC/builtins libraries on `PATH` (Windows) or `LD_LIBRARY_PATH` (Linux). |
| TensorRT | `anr_tensorrt_native.dll` (Linux: `libanr_tensorrt_native.so`) from the Cargo build's `OUT_DIR`, plus compatible TensorRT/CUDA runtime libraries and driver. `ANR_TENSORRT_SHIM` overrides the shim **file**; `ANR_TENSORRT_RUNTIME` accepts a TensorRT DLL directory or library file. Also configure the vendor-library search path, including NVRTC/builtins. |
| OpenVINO | `anr_openvino_native.dll` (`libanr_openvino_native.so` / `.dylib`) from `OUT_DIR`, plus matching OpenVINO Runtime, ONNX frontend, chosen device plugin, and their dependencies. `ANR_OPENVINO_SHIM` overrides the shim **file**; `ANR_OPENVINO_RUNTIME` points to the Runtime library **file**, not a directory. Preserve the vendor runtime/plugin layout and configure the platform library search path. |
| OpenCL | System OpenCL ICD loader and a vendor hardware-GPU ICD/runtime/driver supporting OpenCL 1.2+. A loader alone is not a GPU implementation. |

On Windows, the TensorRT loader matches the DLL names imported by the shim. Its default search order is the shim directory, the executable directory, existing absolute `PATH` directories in order, then the recorded SDK directories, with duplicates removed across stages. Empty/relative `PATH` entries do not enable current-working-directory searches. Explicit `ANR_TENSORRT_RUNTIME` and `CUDA_ROOT` / `CUDA_PATH` overrides always take precedence and must provide their matching runtime family; an invalid override is an error, not permission to select another directory or installed version. The loader does not rewrite the process `PATH`.

Keep shims beside the executable or set their explicit overrides. Merely copying `anr` does not deploy these dependencies, and Cargo installation does not automatically collect the private shims or vendor libraries. Check each vendor's redistribution terms in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md); driver/SDK redistribution is not granted by this project's license. Default release ZIPs remain CPU ORT packages and do not bundle these SDKs or native runtimes.

To exercise real CLI paths after configuring the SDKs, runtimes, hardware, and `ANR_DFN3_MODEL`, run `cargo test -p anr-cli --release --no-default-features --features cuda,tensorrt,openvino,opencl --test cli compute_ -- --include-ignored`, selecting only available features. Recorded Windows x64 acceptance completed with exit 0, 10 passed and 0 ignored, covering raw/WAV, independent stereo state, exact tail length, strength, and batch reset. CUDA/TensorRT/OpenCL ran on RTX 2080 Ti; OpenVINO ran explicitly on AMD Ryzen 7 5800X CPU. SDK/runtime versions were CUDA 13.3, TensorRT 10.16.1, and OpenVINO 2026.4.0. All four also passed the streaming reference. OpenVINO GPU compilation failed on the original graph's `NonZero_1_count` / `count_nonzero` kernel selection; NPU was not tested. These are not Linux/macOS GPU, other-device, sound-quality, or real-time guarantees.


### What has been tested

CI is configured for Windows x64 (`windows-2025`), Linux x64 (`ubuntu-24.04`), macOS arm64 (`macos-15`), and macOS x64 (`macos-15-intel`), using Rust stable. SDK-free workspace gates exclude `anr-tensorrt` and `anr-openvino`; explicit feature sets avoid accidentally enabling SDK-dependent features through `--all-features`. The workflow covers ordinary tests, the explicit CPU ORT real-model test, no-backend and native-only non-GPU CLI tests, lint/compile/doc checks, and extracted-ZIP CPU inference. SDK builds and hardware-dependent native tests require separate configured environments. This is a description of the workflow, not evidence that hosted jobs have passed. Runner labels are documented by [GitHub](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

The small ONNX control graphs check processing, state, and stream formats. They are not DFN3 weights and do not establish real-model sound quality. Record real-model results, compressed format variants, optional providers, and minimum OS/runtime requirements separately. A Linux GNU build does not establish support for musl or arbitrarily old glibc. The tool does not provide real-time audio devices, virtual microphones, drivers, or a cloud service.

Recorded local verification used Rust 1.99; that records the environment, not a fixed MSRV:

| Environment | Executed checks | Not established by these checks |
| --- | --- | --- |
| Windows x64 / RTX 2080 Ti | Real-model ORT inference; GL/GLES/Vulkan/wgpu/D3D12 against the independent 61-frame reference; CLI raw/WAV, stereo isolation, tail length, strength, and batch reset. | Other GPUs' behavior, sound-quality improvement, or real-time performance. |
| WSL Ubuntu 24.04 x64 | ORT inference and CPU ZIP smoke checks; compute-only compilation and non-GPU tests, including GLES-only tests and Clippy. | Linux GPU execution or support for other Linux distributions/libc versions. |
| macOS arm64/x64 | Native-runner workflow configuration only. | Local native, packaging/signing, or GPU results. |

Earlier installed-release checks covered unrelated working directories, model/runtime relocation, raw and unknown-length WAV stdin, closed stdout, cancellation, and protected file aliases. These are recorded coverage, not current test-count promises. Real minimal runtimes, ORT GPU/NPU providers, and hosted CI jobs remain unverified locally.

### License and redistribution

The project code is **MIT OR Apache-2.0**; choose either. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE). Including both full texts in each standalone source package is this project's distribution choice, not a Cargo requirement.

Dependencies and model weights keep their own licenses. The supplied model asset and its matching manifest are explicitly **MIT OR Apache-2.0** as authorized by the maintainer; the ZIP includes `models/LICENSE-MIT`, `models/LICENSE-APACHE`, and `models/NOTICE.md`. This does not grant rights to unrelated model exports. Symphonia and the ORT adapter's `onnx-protobuf` use MPL-2.0; rubato, Rust `ort`, ONNX Runtime, and optional providers have separate terms. See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for source availability, modification, and binary redistribution obligations.

## 简体中文

`anr-cli` 提供命令 `anr`，用于音频文件降噪、逐个文件的批处理，以及 stdin/stdout 音频管道。它通过 `anr` 调用 DFN3，默认使用 ONNX Runtime，也可选择不依赖 ORT 的独立 OpenGL、OpenGL ES、Vulkan、wgpu、Direct3D 12 Compute、CUDA、TensorRT、OpenVINO 和 OpenCL 后端。CLI 自行完成解码、带限重采样和多声道处理，运行时不需要 FFmpeg 或 Python。

源码构建需要 **最新 Rust stable 工具链**。默认 features 会在构建时下载并准备固定 DFN3 模型，`cargo install` 也适用；应用 ZIP 附带同一模型和 CPU 运行库。`--model` 可省略：显式路径优先，否则先查找可执行文件旁的 `models/dfn3.onnx`，再使用构建时记录的模型源文件，不假定源码目录或当前工作目录。ORT 后端只接受下表三个输入和三个输出，均为 `f32`：

| 方向 | 名称 | 运行时的大小 |
| --- | --- | --- |
| 输入 | `input_frame` | `[480]` |
| 输入 | `states` | `[45304]` |
| 输入 | `atten_lim_db` | `[]` 或 `[1]` |
| 输出 | `enhanced_audio_frame` | `[480]` |
| 输出 | `new_states` | `[45304]` |
| 输出 | `lsnr` | `[]` 或 `[1]` |

一维长度可用符号维度声明，但运行时长度不能改变。这些是 ORT 的端口检查，不是 Compute 后端的兼容性判断；Compute 另要求后文指定的精确 SHA-256。模型授权、音质和算法延迟需分别核实。

### 安装与第一次运行

#### 使用应用 ZIP

Release workflow 在平台检查通过后，为 Windows x64、Linux x64、macOS Apple Silicon 和 macOS Intel 生成 `anr-<版本或提交>-<target>.zip`。ZIP 包含可执行文件、CPU ONNX Runtime **1.28.0**、`models/dfn3.onnx` 和适用许可，处理音频不需要 Rust、Python 或 FFmpeg。[模型 Release](https://github.com/lona-cn/anr/releases/tag/model-dfn3-fe5eb64fa2e4) 是独立模型资产，不是 CLI 安装包。

完整解压，进入解压目录后运行：

```sh
./anr denoise input.wav --output clean.wav
```

Windows 使用 `.\anr.exe`。模型从可执行文件旁自动查找，不受当前工作目录影响；随包共享库需与可执行文件保持同目录。默认不需设置 `ORT_DYLIB_PATH`，若以前设置过它，想用随包运行库时应先移除旧的覆盖值。

Windows 可能需要安装微软 [Visual C++ x64 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)。Linux 包在 Ubuntu 24.04 构建，使用 glibc，不等于支持 musl 或任意旧 glibc。macOS 包使用 ad-hoc 签名，未公证；下载后按系统提示允许打开。不包含 GPU provider。使用前核对 Release 附带的 `SHA256SUMS`。

#### 从源码构建

在源码工作区根目录安装。默认构建方式使用 ORT 提供的预编译库；**Intel macOS 请改用后面的自备运行库方式。**

```text
cargo install --path crates/anr-cli --locked
anr --help
anr backends --json
```

二进制名是 `anr`，不是 `anr-cli`。把安装目录加入 `PATH`，或用完整路径运行可执行文件。从 crates.io 安装 `0.1.1` 可使用 `cargo install anr-cli --version 0.1.1 --locked`。

默认 features 为 `ort-download-binaries`、`download-models` 和 `copy-models`：启用 ORT 后端，下载预编译 CPU 运行库和固定模型，并将模型复制到构建产物旁。首次构建需要网络、目标工具链和平台支持：

- Windows：MSVC C++ 工具链和 Windows SDK。
- Linux：C/C++ 链接工具链、`pkg-config` 和 OpenSSL 开发文件，例如 Ubuntu 的 `libssl-dev`。
- macOS：Xcode Command Line Tools。

当前 `ort 2.0.0-rc.15` 的默认预编译运行库为 1.31.0，不包含 Intel macOS。该平台需自行构建，或使用兼容 API 27 的共享库，包括 1.28.0；只安装 Rust target 不够。

例如，在 Intel macOS 上准备好兼容的 CPU `libonnxruntime.dylib` 后，在运行 `anr` 的环境里设置实际绝对路径，再安装动态加载版本：

```sh
export ORT_DYLIB_PATH="/absolute/path/to/libonnxruntime.dylib"
cargo install --path crates/anr-cli --locked --no-default-features --features ort-load-dynamic,download-models,copy-models
anr backends --json
```

把路径换成真实文件，并在以后运行时保留设置。`backends` 只列出编译时加入的支持，实际执行 `denoise` 才会加载并使用运行库。可按 [ONNX Runtime CPU 源码构建说明](https://onnxruntime.ai/docs/build/inferencing.html) 准备运行库；CI 固定构建 [微软 v1.28.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0)，不假设该 release 提供 Intel macOS 预编译包。此时 CMake、Python 等只用于源码构建，不是安装后 CLI 处理音频的依赖。

处理单文件、批处理和信息查询：

```text
anr denoise voice.wav --output clean.wav
anr denoise a.flac b.mp3 --output-dir cleaned
anr inspect voice.wav
anr inspect voice.wav --json
```

需要其他模型时使用 `--model /path/to/dfn3.onnx`，替换为自己有权使用的实际路径；Windows 同样可用带引号的本机路径。显式相对路径以启动命令时的工作目录为准，自动查找则先以可执行文件位置为准，再使用构建时记录的路径。`inspect` 不加载模型；`backends` 只说明编译时加入了哪些支持，不证明运行库或 GPU 已能使用。

`download-models` 将通过摘要校验的模型缓存到 `$CARGO_HOME/anr/models/dfn3-fe5eb64fa2e4/`，默认 Cargo home 为 `~/.cargo`。`copy-models` 独立负责复制，在 Cargo profile、示例和测试目录下生成 `models/dfn3.onnx`。**构建前**将 `ANR_MODEL_PATH` 设为本地文件的绝对路径，即可跳过模型下载，或单独使用 `copy-models`。仅下载的构建和 `cargo install` 可使用持久源路径，不要求程序旁已有副本。向其他机器分发时应附带模型及适用许可、通知，不能依赖原构建机器的缓存。运行时不下载模型。

关闭 default features 可去掉自动模型准备及下载器依赖，之后需传入 `--model`、随程序提供模型，或按需重新启用模型 features。缓存缺失且设置 `CARGO_NET_OFFLINE=true` 时会报错而非联网；离线构建可设置 `ANR_MODEL_PATH`。详情见 [模型准备说明](../anr/README.md#模型下载与复制)。

### 参数与批处理

```text
anr denoise INPUT... --output PATH [--model MODEL] [OPTIONS]
anr denoise INPUT... --output-dir DIR [--model MODEL] [OPTIONS]
anr inspect INPUT [--json]
anr backends [--json]
```

| 参数 | 含义 |
| --- | --- |
| `INPUT...` | 明确列出的文件路径；单输入时可用 `-` 表示 stdin。程序不递归扫描目录，也不自行展开 glob。 |
| `--output PATH` | 单输入的目标文件；`-` 表示 stdout。与 `--output-dir` 互斥。 |
| `--output-dir DIR` | 逐个处理文件，按输入文件名去掉扩展名后生成 `.wav`，明确选择 raw 时生成 `.f32le`。同名结果不自动改名。 |
| `--model PATH` | 可选的模型覆盖路径；省略时使用随程序或构建时准备的模型。显式路径无效时直接报错，不改用默认模型。 |
| `--backend ort\|gl\|gles\|vk\|wgpu\|dx\|cuda\|tensorrt\|openvino\|opencl` | 默认 `ort`。各原生后端需要主动启用对应 Cargo feature，并使用后文说明的固定模型。 |
| `--provider cpu` | 仅适用于 ORT，ORT 未传时默认 `cpu`；其他 provider 需对应 feature 和运行环境。任何非 ORT 后端与显式 `--provider` 同用均为参数错误。 |
| `--openvino-device CPU\|GPU\|GPU.n\|NPU` | 仅适用于 OpenVINO，省略时明确使用 `CPU`；`GPU.n` 指定 GPU 索引，例如 `GPU.0`。GPU/NPU 失败不切换 CPU。与其他后端同用为参数错误。 |
| `--strength 80` | 默认 80；必须是 `0..=100` 的有限数值，拒绝 NaN、无穷和越界值，不静默修正。 |
| `--input-format auto\|f32le` | 默认探测带头音频；raw 必须明确选 `f32le`。 |
| `--input-rate HZ` / `--input-channels N` | raw 输入必填的正数；带头音频不能用它们覆盖真实规格。 |
| `--output-format wav\|f32le` | 文件可由 `.wav` / `.f32le` 扩展名识别；其他扩展名须明确指定。stdout 必须明确选 `f32le`。 |
| `--output-rate HZ` | 可选正采样率；默认保留输入采样率。 |
| `--wav-format pcm16\|pcm24\|float32` | 默认 PCM16；raw 输出不能显式指定此参数。 |
| `--overwrite` | 允许处理完成后替换已有目标，仍禁止覆盖输入、模型或本批次已经成功保存的结果。 |
| `--quiet` / `--verbose` | 互斥的全局选项，控制 stderr 诊断，不改变音频 stdout。 |

批处理在开始输出前检查全局参数、模型/后端（ORT 还检查 provider）和可确定的目标冲突，并在处理前及提交前复核实际文件身份，包括大小写不敏感文件系统的别名。如果先前不存在的别名在本批次较早的结果保存后才显现，后一个文件会失败，不会覆盖先前结果。每个声道持有独立 reducer，文件间复用并重置状态；失败文件清理临时输出并继续，最终退出 1，成功文件保留。若文件已经保存但无法核验其身份，会明确报告已保存并停止后续处理；不撤销整批结果。

文件先写到目标同目录的临时文件，推理、编码和 WAV 文件收尾全部完成后才移到最终路径。默认不覆盖；即使使用 `--overwrite`，也不会在开始处理前截断旧文件。这不保证断电后数据仍然完整。stdout 已发出的字节无法撤回；失败时必须检查退出码和 stderr，不能把部分音频流当完整结果。

### 音频格式和时长

带头输入使用 Symphonia 0.5.5 的解码器和容器探测，启用其 `all` features，包括 WAV/PCM、FLAC、MP3、Ogg/Vorbis、AAC-LC/M4A 等支持。**容器可识别，不代表里面所有 codec/profile 都能解码。** 例如 Opus 不在当前 codec 集合内。具体文件仍以解码结果为准；这里没有声称所有编码变体或平台都已实测通过。

多音轨默认选择第一个可解码音轨；`inspect` 显示所选音轨、采样率、声道等信息。JSON 字段为 `format`、`codec`、`track_id`、`sample_rate`、`channels`、`frames`、`duration_seconds`；未知帧数或时长为 `null`。处理过程中改变采样率或声道数会报错。不可 seek 的编码 stdin 还受 demuxer 限制，需要 seek 的输入可能失败；CLI 不会因此缓存整条输入或暗中写入磁盘。

对有可靠声明帧数的固定宽度 WAV，按实际解码帧数校验完整性；提前 EOF 会报错，不能提交短音频替换旧目标。未知长度 WAV 标记仍允许流式读取，压缩格式的估算时长不作为精确校验预算。这不表示能够发现每一种容器或 codec 损坏。

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

### 编入不同后端与执行 provider

```text
cargo build -p anr-cli --release
cargo build -p anr-cli --no-default-features
cargo build -p anr-cli --features ort-directml
cargo build -p anr-cli --no-default-features --features ort-load-dynamic
cargo build -p anr-cli --no-default-features --features gles
cargo build -p anr-cli --no-default-features --features gl,gles,vk,wgpu,dx
```

完全不启用 features 时，可以列能力和检查音频，但不能降噪。`ort` 只启用后端，通过 `ORT_LIB_LOCATION` 链接自备运行库；默认 `ort-download-binaries` 额外负责 CPU 库下载。`--no-default-features --features ort-load-dynamic` 改为加载自备共享库，不带下载器或 TLS/OpenSSL 构建依赖。`ort-cuda`、`ort-tensorrt`、`ort-directml`、`ort-coreml`、`ort-xnnpack` 转发到 `anr` 的 ORT provider，需要对应平台、库和驱动；选择 feature 不会安装它们。这些 feature 与原生 `cuda`、`tensorrt` 不同。

例如，在 Windows 编入 DirectML 后选择 `--provider directml`，在 macOS 编入 CoreML 后选择 `--provider coreml`。默认仍为 `cpu`。项目正常允许 ORT 在 CUDA、TensorRT、DirectML、CoreML 之外，将不支持或按自身分配规则更适合 CPU 的算子交给 CPU。这是默认策略，不是临时选项；选择 GPU provider 不代表所有算子都在 GPU 上执行。明确指定的 provider 注册或初始化失败仍报错，不会把不可用的 provider 静默换成 CPU。XNNPACK 本身是 CPU provider，也允许普通 CPU kernel 执行算子。`backends --json` 中，`compiled` 只表示编译时加入了支持；`runtime_available: null` 表示尚未检查运行时可用性，不代表模型成功运行。

ORT 的 CUDA/TensorRT provider 会在创建各个 Session 前自动准备内存中的模型，与 benchmark 共用 `anr-ort` 加载入口。不需要额外参数、手动生成转换模型或安装 Python；原 ONNX 和权重不变，衰减量仍可调整。CPU、DirectML 及原生后端保留原有加载方式。


动态构建需在启动前将 `ORT_DYLIB_PATH` 指向兼容 API 27 的共享库**文件**，不是目录；加载失败在创建输出前正常返回退出码 1。CLI 为整批文件准备一个环境，库使用者则显式持有环境。共享链接可配置 `ORT_LIB_LOCATION`、`ORT_PREFER_DYNAMIC_LINK=1` 和系统库搜索路径。分发时提供必需运行库及通知，不能认为所有构建方式都只需复制可执行文件。

可选 `gl`、`gles`、`vk`、`wgpu`、`dx`、`cuda`、`tensorrt`、`openvino`、`opencl` features 启用原生后端，不会启用 ONNX Runtime。用 `--no-default-features` 去掉 ORT，再选择环境支持的 features。即使没有编入 ORT，`--backend` 默认值仍是 `ort`，因此 native-only 构建必须明确指定后端；不能传 `--provider`，包括 `--provider cpu`。

从源码运行 GLES-only 版本时，使用 Cargo 直接执行刚构建的程序，避免调用 `PATH` 中旧的 `anr`：

```text
cargo run -p anr-cli --release --no-default-features --features gles -- denoise input.wav --model /path/to/dfn3.onnx --backend gles --output clean.wav
```

GL/GLES/Vulkan/wgpu/D3D12 都只将原 ONNX 用作参数容器，不执行通用 ONNX 图。固定模型 SHA-256 必须为 `fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9`，即使端口大小一致，其他导出版本也会拒绝。通过 `anr-dfn3` 共用固定执行顺序和 WGSL 算子，FFT、神经网络及音频合成都在 GPU 执行；保留 48 kHz / 480 采样分帧、CLI 重采样、声道独立状态、strength、reset 和精确输出长度。强度为零保留原衰减分支，不表示原音旁路。

`gles` 需要 Windows/WGL 或 Linux 硬件 EGL/pbuffer 创建真实 ES 3.1+ context，复用 `anr-gl` 执行器，生成 GLSL ES 310；在 fence 完成后映射输出帧，不调用桌面专用 `glGetBufferSubData`，也不接受桌面 GL 的 ES 兼容模式。本后端未提供 Android、WebGL 或 macOS context 创建。

`gl` 需要 Windows 桌面 GL 4.3 或 Linux 硬件 EGL 的桌面 GL 4.3/pbuffer；`vk` 需要 Windows/Linux 硬件 Vulkan 1.1；`wgpu` 在 Windows/Linux/macOS 使用原生硬件 Vulkan/DX12/Metal adapter；`dx` 使用 Windows 硬件 D3D12 与系统 D3DCompiler 47。原生 VK/DX 不经 wgpu，DX 不是 DirectML。不静默选择 CPU 或软件渲染；硬件不支持或初始化失败会报错。macOS GPU 路径尚未验证。

原生后端不能传 `--provider`，包括 `--provider cpu`。`backends --json` 列出十个名称，但不加载运行库或创建 GPU device；原生项没有 ORT provider，`runtime_available` 保持 `null`，只表示没有探测可用性，不代表运行库或设备已就绪。verbose 日志显示所选 backend，仅 ORT 显示 provider。

真实模型 CLI 测试需主动启用：将 `ANR_DFN3_MODEL` 指向模型，执行 `cargo test -p anr-cli --no-default-features --features gl,gles,vk,wgpu,dx --test cli compute_ -- --include-ignored`，按机器支持情况选择 features。测试覆盖 raw/WAV、立体声独立状态、非整帧长度、批处理重置和 strength。每个启用的 Compute 后端都需要真实硬件，不能把 ignored 当作硬件测试通过。

### 原生 CUDA、TensorRT、OpenVINO 和 OpenCL 部署

四种新增后端均需主动启用，并使用前述固定 SHA-256 模型，保留 CLI 重采样、声道独立状态、strength、reset、尾帧处理和精确输出长度。CUDA 和 OpenCL 在 GPU 执行完整固定 FP32 DSP/FFT/神经网络计划；CUDA 使用 Driver/NVRTC 与 replay graph，OpenCL 要求 OpenCL 1.2+ 硬件 GPU ICD，不接受 CPU 实现。TensorRT 使用两个真实 standard-layer 神经网络 engine 与 CUDA DSP/FFT graph，强类型 FP32、禁用 TF32，不经 ORT、ONNX parser 或推理 plugin。目前没有持久 engine cache；已记录的一次初始化为 17.342 秒，不能承诺快速冷启动。

OpenVINO 通过原生 Runtime 直接导入原 ONNX，将循环状态转为 stateful，每条流使用独立 InferRequest。默认明确使用 CPU，不选 AUTO/HETERO。`--openvino-device GPU`、`GPU.0` 或 `NPU` 只请求对应设备；设备缺失、算子不支持均报错，不静默回退 CPU。NPU 运算精度取决于设备，不能因模型为 FP32 就宣称 NPU 全程 FP32。

CUDA/OpenCL 的 Rust crate 不需要构建时 vendor SDK，但运行仍需兼容库与驱动。TensorRT/OpenVINO 构建需要现有 SDK 和 C++17 编译器，`build.rs` 编译私有 shared shim，Rust 通过 `libloading` 加载；不会系统安装或自动下载 SDK。

```text
cargo build -p anr-cli --release --no-default-features --features cuda,opencl
cargo run -p anr-cli --release --no-default-features --features cuda -- denoise input.wav --model /path/to/dfn3.onnx --backend cuda --output clean.wav
```

启用 SDK 后端前，在构建环境配置真实 SDK 路径。PowerShell 示例（须替换路径）：

```powershell
$env:TENSORRT_ROOT = "C:\sdk\TensorRT"
$env:CUDA_ROOT = "C:\sdk\CUDA"
$env:OPENVINO_ROOT = "C:\sdk\openvino"
cargo build -p anr-cli --release --no-default-features --features cuda,tensorrt,openvino,opencl
```

TensorRT 需要 `TENSORRT_ROOT`、`CUDA_ROOT`（或 `CUDA_PATH`）下的 headers 和链接库，可用 `TENSORRT_LIB_DIR`、`CUDA_LIB_DIR` 覆盖链接库目录。OpenVINO 支持 `OPENVINO_ROOT` 或指向 SDK CMake 布局的 `OpenVINO_DIR`；官方 wheel 中含 `include/`、`libs/` 的 `openvino` 目录也可作为 SDK，Python 不是产品运行依赖。

构建后分别使用 `--backend tensorrt`、`--backend opencl` 或 `--backend openvino --openvino-device CPU`。相对地，ORT CUDA 使用 `ort-cuda` feature 和 `--backend ort --provider cuda`；ORT TensorRT 使用 `ort-tensorrt` 和 `--backend ort --provider tensorrt`。原生后端不继承 ORT 的算子级 CPU 分配策略。

自行分发应用时，除模型和通知外，还需部署以下依赖，不能依赖构建机器记录的绝对路径：

| 后端 | 运行时部署要求 |
| --- | --- |
| CUDA | NVIDIA Driver API 12.0+，NVRTC major 12 或 13（优先 13），驱动还须能加载所选 NVRTC 产生的 PTX；仅支持 Driver API 12.0 不代表兼容 NVRTC 13 PTX。Windows 加载 `nvrtc64_130_0.dll` / `nvrtc64_120_0.dll`，Linux 加载 `libnvrtc.so.13` / `.12` / 无版本名库。将匹配的 NVRTC/builtins 库目录加入 Windows `PATH` 或 Linux `LD_LIBRARY_PATH`。 |
| TensorRT | Cargo 构建 `OUT_DIR` 中的 `anr_tensorrt_native.dll`（Linux 为 `libanr_tensorrt_native.so`），以及兼容 TensorRT/CUDA 运行库、驱动。`ANR_TENSORRT_SHIM` 覆盖 shim **文件**；`ANR_TENSORRT_RUNTIME` 可设为 TensorRT DLL 目录或库文件。还需配置 vendor 库搜索路径，包括 NVRTC/builtins。 |
| OpenVINO | `OUT_DIR` 中的 `anr_openvino_native.dll`（`libanr_openvino_native.so` / `.dylib`），以及匹配的 OpenVINO Runtime、ONNX frontend、所选设备 plugin 及其依赖。`ANR_OPENVINO_SHIM` 覆盖 shim **文件**；`ANR_OPENVINO_RUNTIME` 指向 Runtime 库**文件**，不是目录。保留 vendor runtime/plugin 布局并配置平台库搜索路径。 |
| OpenCL | 系统 OpenCL ICD loader，以及支持 OpenCL 1.2+ 的厂商硬件 GPU ICD/runtime/driver。只有 loader 不等于拥有 GPU 实现。 |

Windows 上，TensorRT loader 按 shim 实际导入的 DLL 名，默认依次搜索 shim 所在目录、executable 所在目录、`PATH` 中存在的绝对目录，最后查构建时 SDK 目录，各阶段去重。空/相对 `PATH` 项不会启用当前工作目录搜索。显式设置的 `ANR_TENSORRT_RUNTIME` 和 `CUDA_ROOT` / `CUDA_PATH` 始终优先，必须提供各自对应的运行库；覆盖路径无效会报错，不会改选其他目录或已安装版本。loader 不改写进程 `PATH`。

shim 应放在 executable 旁或显式设置覆盖路径。只复制 `anr` 不足以部署这些依赖，Cargo 安装也不会自动收集私有 shim 或 vendor 库。再分发权限须按 [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) 中各厂商条款核实；项目许可不授予任意驱动/SDK 的再分发权。默认 Release ZIP 仍为 CPU ORT 包，不附带这些 SDK 或原生运行库。

配置 SDK、运行库、硬件及 `ANR_DFN3_MODEL` 后，可执行 `cargo test -p anr-cli --release --no-default-features --features cuda,tensorrt,openvino,opencl --test cli compute_ -- --include-ignored` 验收真实 CLI，features 只选本机可用项。已记录 Windows x64 验收退出 0，10 passed、0 ignored，覆盖 raw/WAV、立体声独立状态、精确尾帧长度、strength 与批处理 reset。CUDA/TensorRT/OpenCL 使用 RTX 2080 Ti，OpenVINO 明确使用 AMD Ryzen 7 5800X CPU；SDK/runtime 版本为 CUDA 13.3、TensorRT 10.16.1、OpenVINO 2026.4.0。四后端另已通过流式 reference。OpenVINO GPU 在原图 `NonZero_1_count` / `count_nonzero` kernel 选择处编译失败，NPU 未测；不能据此宣称 Linux/macOS GPU、其他设备、音质或实时性能已验证。


### 哪些已经测过

CI 为 Windows x64（`windows-2025`）、Linux x64（`ubuntu-24.04`）、macOS arm64（`macos-15`）和 macOS x64（`macos-15-intel`）配置 Rust stable 检查。SDK-free 工作区检查排除 `anr-tensorrt`、`anr-openvino`，使用明确 feature 组合，避免 `--all-features` 意外编入 SDK 后端。工作流覆盖普通测试、CPU ORT 真实模型测试、无后端和 native-only 非 GPU CLI 测试、lint/编译/文档检查及解压 ZIP 后的 CPU 推理；SDK 构建和原生硬件测试须在另行配置的环境验收。这是配置说明，不是 hosted job 已通过的证据。runner 标签来源见 [GitHub 官方文档](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)。

自制 ONNX 控制图只验证处理、状态和音频流格式，不是 DFN3 权重，也不证明真实模型音质。真实模型、压缩格式变体、可选 provider 及最低 OS/运行库要求，需要分别记录实际结果。Linux GNU 构建不代表支持 musl 或任意旧 glibc。本工具不提供实时音频设备、虚拟麦克风、驱动或云服务。

以下本地记录使用 Rust 1.99，只说明验收环境，不表示固定 MSRV：

| 环境 | 已执行的检查 | 不能据此确定的事项 |
| --- | --- | --- |
| Windows x64 / RTX 2080 Ti | ORT 真实模型；GL/GLES/Vulkan/wgpu/D3D12 的独立 61 帧数值基准；CLI raw/WAV、立体声独立状态、尾帧长度、strength 和批处理 reset。 | 其他 GPU 的行为、音质改善或实时性能。 |
| WSL Ubuntu 24.04 x64 | ORT 推理与 CPU ZIP smoke；Compute-only 编译和非 GPU 测试，包含 GLES-only 测试及 Clippy。 | Linux GPU 执行，以及其他发行版/libc 的兼容性。 |
| macOS arm64/x64 | 仅配置原生 runner 工作流。 | 本地原生运行、打包/签名或 GPU 结果。 |

此前安装后验收覆盖无关工作目录、模型/运行库重定位、raw 和未知长度 WAV stdin、提前关闭 stdout、取消以及文件别名保护。这些是已记录的覆盖范围，不是当前测试数量保证。真实 minimal runtime、ORT GPU/NPU provider 及 hosted CI jobs 仍未在本地验证。

### 许可与再分发

项目代码采用 **MIT OR Apache-2.0**，任选其一；见 [LICENSE-MIT](LICENSE-MIT) 和 [LICENSE-APACHE](LICENSE-APACHE)。在独立源码包内附带两份完整文本是本项目的分发选择，不是 Cargo 的强制要求。

依赖和权重保留各自的许可。本项目提供的模型资产及对应清单已由维护者授权按 **MIT OR Apache-2.0** 分发，ZIP 内有 `models/LICENSE-MIT`、`models/LICENSE-APACHE` 和 `models/NOTICE.md`；不能将这一授权泛化到其他模型。Symphonia 与 ORT 适配器使用的 `onnx-protobuf` 为 MPL-2.0；rubato、Rust `ort`、ONNX Runtime 及可选 provider 另有条款。源码获取、修改及二进制分发义务见 [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)。
