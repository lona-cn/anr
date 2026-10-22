# anr

[English](#english) | [简体中文](#简体中文)

## English

`anr` provides a Rust interface to DFN3 backends: ORT, OpenGL, GLES, Vulkan, wgpu, D3D12, CUDA, TensorRT, OpenVINO, and OpenCL. It processes **48 kHz, mono `f32` samples** with equally sized input and output. No backend is enabled by default. With `ort`, use `anr::DFN3`; with another backend, use its module, such as `anr::cuda::DFN3`.

The library does not read microphones, decode WAV, or resample audio. You'll need the **latest stable Rust toolchain**. Default features download and prepare the supported DFN3 model at build time.

### Choose a backend

| Cargo feature | Backend | Type |
| --- | --- | --- |
| `ort` | ONNX Runtime | `anr::DFN3` |
| `gl` | OpenGL | `anr::gl::DFN3` |
| `gles` | OpenGL ES | `anr::gles::DFN3` |
| `vk` | Vulkan | `anr::vk::DFN3` |
| `wgpu` | wgpu | `anr::wgpu::DFN3` |
| `dx` | Direct3D 12 | `anr::dx::DFN3` |
| `cuda` | Native CUDA Driver/NVRTC | `anr::cuda::DFN3` |
| `tensorrt` | Native TensorRT + CUDA DSP/FFT | `anr::tensorrt::DFN3` |
| `openvino` | Native OpenVINO Runtime, explicit CPU/GPU/NPU | `anr::openvino::DFN3`, `anr::openvino::Device` |
| `opencl` | Native OpenCL 1.2 GPU | `anr::opencl::DFN3` |

To build without ORT, set `default-features = false` and list the backend you need under `features`. For example:

```toml
[dependencies]
anr = { version = "0.1.1", default-features = false, features = ["wgpu"] }
```

List the backends your program needs under `features`. Cargo includes all of them; your program must choose which one to use. The non-ORT backends support only the project's specified DFN3 model, not arbitrary ONNX files. Hardware and system requirements are below. Native `cuda` / `tensorrt` are independent implementations, not ORT's `ort-cuda` / `ort-tensorrt` providers.

### ORT quickstart

```toml
[dependencies]
anr = { version = "0.1.1", features = ["ort"] }
```

The dependency examples use version `0.1.1`. To use a local checkout, replace the version dependency with a `path` dependency pointing to the crate directory.

```rust,no_run
use anr::{DFN3, NoiseReduction, Result};

fn main() -> Result<()> {
    anr::ort::initialize_runtime()?;
    let environment = anr::ort::init().build().map_err(anr::Error::backend)?;
    let mut denoiser = DFN3::from_file(&environment, anr::default_model_path()?)?;
    denoiser.set_factor(80.0);

    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    denoiser.run(DFN3::SAMPLE_RATE, &input, &mut output)?;
    denoiser.reset();
    Ok(())
}
```

The example processes silence; replace it with your own samples. `default_model_path()` finds `models/dfn3.onnx` beside the executable, then the model prepared at build time; it does not depend on the working directory or download at runtime.

### Model download and copying

`download-models` and `copy-models` are both enabled by default and do not enable an inference backend. The downloader fetches [the pinned DFN3 release](https://github.com/lona-cn/anr/releases/download/model-dfn3-fe5eb64fa2e4/dfn3.onnx), verifies SHA-256 `fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9`, and reuses `$CARGO_HOME/anr/models/dfn3-fe5eb64fa2e4/dfn3.onnx` (`~/.cargo` when `CARGO_HOME` is unset).

`copy-models` copies the selected file to `models/dfn3.onnx` in the Cargo profile directory and its `examples` and `deps` directories, including custom target directories and cross-compilation layouts. It does not imply `download-models`. Set **`ANR_MODEL_PATH` to an absolute local file path before building** to skip downloading, or to use `copy-models` on its own. Changes to that source trigger another copy.

With only `download-models`, `default_model_path()` uses the cached file without copying. `cargo install` can also use this persistent cache after its temporary build directory is removed. To distribute a program to another machine, ship `models/dfn3.onnx` beside it and include the model's licenses/notices; the build-cache path is not portable. The default downloader uses native TLS, like the ORT downloader.

Use `default-features = false` to omit automatic model preparation and the downloader's dependencies. You can still pass an explicit model path or bundle a file beside the executable. A cold cache with `CARGO_NET_OFFLINE=true`, a failed download, or a checksum mismatch fails the build; provide `ANR_MODEL_PATH` for an offline build. Documentation builds on docs.rs skip model preparation.

### Keep one denoiser per stream

Reuse the same `DFN3` for continuous audio, and don't share it between unrelated streams. The model uses information from earlier audio. Call `reset()` before switching recordings or restarting playback; this clears the audio state without changing the strength.

Use chunks that are multiples of 480 samples where possible. `DFN3::FRAME_SIZE = 480`, or 10 ms of audio. `run` accepts any nonzero length, but pads a short final frame with zeros, processes the full frame, and returns only the requested samples. The zeros still affect later state, so many short calls may produce different results from one call over the whole recording.

The library does not compensate for model latency and has no end-of-stream flush operation. Your application must split multichannel audio before processing it.

### Check the input and output

- Supply at least one input sample.
- Use a sample rate of 48,000 Hz.
- Allocate an output slice with the same length as the input.

If any of these checks fail, the output and processing state stay unchanged. An error during model execution may leave both partly updated; it does not undo work already done.

Without a backend feature, `anr` exports `NoiseReduction`, `Error`, `Result`, and `default_model_path`. The default `ort-download-binaries` option does not enable ORT by itself. `NoiseReduction` works as `Box<dyn NoiseReduction>`; `Send` allows moving a reducer to another thread but does not guarantee concurrent calls or nonblocking execution.

### Adjust the strength

All DFN3 implementations expose `set_factor(0..=100)` through `NoiseReduction`, mapped to dB as `round(factor / 100 * 60)`. Only `anr::ort::DFN3` additionally provides `set_atten_lim_db` / `get_atten_lim_db` for direct `0..=60` dB access. Those methods are not part of the trait or compute API.

Out-of-range values are clamped. `NaN` leaves the setting unchanged; positive and negative infinity use the upper and lower bounds respectively.

`set_factor` rounds to whole dB; the ORT-only dB setter preserves fractional values. `factor()` rounds the actual setting back to `0..=100`, so it may not return exactly what you passed in. Strength is an attenuation limit, not a dry/wet mix; zero still runs the supplied model and is not an audio bypass.

### Try the ORT raw audio example

Run this from the project root:

```text
cargo run -p anr --features ort --example denoise -- <input.f32le> <output.f32le>
```

The input is **48 kHz mono raw audio, not WAV**: four bytes per sample, stored as little-endian IEEE 754 `f32`. The output uses the same format. This command requires the source checkout; installing the crate alone does not give you the whole project.

To select a different model, prepend its path before the input and output arguments.

### ORT model requirements

For ORT, `DFN3::from_file(&environment, path)` loads a model file using an application-owned `Environment`; `DFN3::new(session)` uses a configured session. Both require exactly these three inputs and three outputs, all `f32`. Input order does not matter. `[]` is a scalar; `[1]` is a one-element vector. Non-ORT `from_file` constructors take only `path` and require the fixed model hash below; OpenVINO also offers `from_file_with_device`.

| Direction | Name | Runtime shape |
| --- | --- | --- |
| Input | `input_frame` | `[480]` |
| Input | `states` | `[45304]` |
| Input | `atten_lim_db` | `[]` or `[1]` |
| Output | `enhanced_audio_frame` | `[480]` |
| Output | `new_states` | `[45304]` |
| Output | `lsnr` | `[]` or `[1]` |

A one-dimensional tensor may declare a symbolic length. Loading checks the rank and any fixed lengths; symbolic lengths do not change the required 480 audio samples and 45,304 state values at runtime. The constants are `DFN3::SAMPLE_RATE = 48000`, `FRAME_SIZE = 480`, and `STATE_SIZE = 45304`.

Audio and state outputs use preallocated buffers, and ONNX Runtime checks their actual lengths. For `lsnr`, only the declared type and shape are checked; its result is neither read nor returned. Models with the same DeepFilterNet3 name may still use different inputs and outputs, so check more than the name.

### Select features and the ORT runtime

- `ort`: provides `DFN3`. With default features, downloads and copies CPU ONNX Runtime libraries at build time; requires network access and platform support.
- `gl`, `gles`, `vk`, `wgpu`, `dx`: enable the corresponding compute backend and `anr::<backend>::DFN3`, without enabling ORT.
- `cuda`, `tensorrt`, `openvino`, `opencl`: enable independent native backends and their `anr::<backend>::DFN3` modules, without ORT. TensorRT/OpenVINO require build-time C++ SDKs; enabling all features therefore requires those SDKs.
- `ort-download-binaries`: the default, weak runtime-download option. It does not activate `anr-ort` by itself; automatic model preparation is independent of ORT.
- `download-models`: enabled by default; downloads and caches the pinned DFN3 model unless `ANR_MODEL_PATH` supplies a local file.
- `copy-models`: enabled by default; copies the selected model beside build outputs, independently of downloading.
- `ort-load-dynamic`: loads your runtime. Disable defaults to omit the downloader and TLS dependencies. Set `ORT_DYLIB_PATH` to a compatible shared library file before initialization.
- `ort-cuda`, `ort-tensorrt`, `ort-directml`, `ort-coreml`, `ort-xnnpack`: compile support for the corresponding libraries. They do not select a GPU or install drivers or other required libraries.

The project pins `ort = 2.0.0-rc.15` with API 27, without `ndarray`. The default prebuilt runtime is 1.31.0; compatible API-27 libraries such as 1.28.0 also work. `anr::ort` exposes the same ONNX Runtime types.

```toml
anr = { version = "0.1.1", default-features = false, features = ["ort-load-dynamic"] }
```

Call `anr::ort::initialize_runtime()` before `anr::ort::init().build()` to return dynamic-loading failures instead of entering the lazy loader. The helper does not build an environment; your application owns and configures it. Share that `Environment` across denoisers. `DFN3::from_file(&environment, path)` explicitly selects the built-in CPU provider, one intra-op thread, and no spinning. To choose another provider, configure a `SessionBuilder`, call `anr::ort::session_from_file(builder, path, &providers)`, then pass the session to `DFN3::new(session)`.

ORT GPU sessions may use CPU kernels for unsupported or CPU-preferred operators. The library preserves the supplied session's operator placement; CLI and benchmark also allow this by default. Explicit provider registration failures remain errors, not an implicit replacement of an unavailable GPU provider with CPU.

The shared loader prepares CUDA/TensorRT graphs in memory before session creation; it never overwrites the ONNX file or runs conversion per audio frame. CUDA's unsupported sigmoid convolution fusion is expanded; TensorRT additionally lowers eligible static state sequences and safe tensor conditionals. CPU/DirectML and native compute are unchanged. `anr::ort::prepare_model` and `ModelTarget` expose the same preparation for callers who already own model bytes; direct upstream session loading bypasses it.


### Compute backends without ORT

Enable `default-features = false, features = ["gles"]` and construct `anr::gles::DFN3::from_file(path)`, or select `gl`, `vk`, `wgpu`, or `dx` with the corresponding module and feature. Each implements `NoiseReduction`, owns a worker thread and GPU-resident weights/state, and waits synchronously for each output. The common fixed plan and WGSL kernels live in `anr-dfn3`; native GL/GLES/Vulkan/D3D12 use Naga-generated GLSL/GLSL ES/SPIR-V/HLSL. No backend substitutes ORT or CPU inference.

```toml
anr = { version = "0.1.1", default-features = false, features = ["gles"] }
```

```rust,no_run
use anr::{gles::DFN3, NoiseReduction, Result};

fn main() -> Result<()> {
    let mut denoiser = DFN3::from_file("models/dfn3.onnx")?;
    denoiser.set_factor(80.0);
    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    denoiser.run(DFN3::SAMPLE_RATE, &input, &mut output)?;
    denoiser.reset();
    Ok(())
}
```

`gles` requires an actual ES 3.1+ context on Windows/WGL or Linux hardware EGL/pbuffer. It shares the executor in `anr-gl`, but does not fall back to desktop GL. Advanced GL-family callers can explicitly select `GlApi` through `DFN3::from_file_with_api(path, api)`; ordinary `anr::gl::DFN3::from_file` remains desktop-only. No Android, WebGL, or macOS context is provided for GLES.

`gl` requires desktop GL 4.3 on Windows/Linux; `vk` requires Windows/Linux hardware Vulkan 1.1; `wgpu` selects native hardware Vulkan/DX12/Metal on Windows/Linux/macOS; `dx` requires Windows hardware D3D12 and system D3DCompiler 47, without WARP, DirectML, or wgpu. Windows / RTX 2080 Ti has exercised all five; other platforms' GPU paths are not verified.

For these GL/GLES/Vulkan/wgpu/D3D12 shader backends, the original external ONNX file supplies parameters only and must have SHA-256 `fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9`. No model conversion, Python, or external shader compiler is required. Other exports and unsupported drivers fail explicitly. Strength zero preserves the model's attenuation branch; it is not an audio bypass.

### Native CUDA, TensorRT, OpenVINO, and OpenCL

These backends validate the same fixed model hash and implement `NoiseReduction`, with one worker per stream and synchronous `run`. CUDA and OpenCL execute the complete fixed FP32 schedule, including FFT, normalization, recurrent inference, spectral filtering, and overlap-add; weights and state stay on GPU. CUDA uses the Driver API, NVRTC compilation, and replay graphs. OpenCL requires a hardware GPU ICD supporting OpenCL 1.2, with no CPU-device fallback. Neither requires a build-time SDK.

TensorRT uses two real standard-layer neural-network engines plus CUDA DSP/FFT graphs, not ORT, an ONNX parser, or a monolithic inference plugin. It uses strongly typed FP32 with TF32 disabled. Initialization builds the engines; there is no persistent engine cache. The recorded initialization cost was 17.342 s, not a fast-cold-start guarantee.

OpenVINO directly imports the original ONNX with its native frontend, rather than reading ONNX only as parameters. It uses `MakeStateful` (`ReadValue`/`Assign`) and one `InferRequest` per reducer; state remains inside the selected plugin, not a host round-trip every frame. `from_file` selects **CPU explicitly**, not as GPU fallback. CPU/GPU request accuracy-oriented FP32 inference; NPU hardware may use FP16 and is not verified against the FP32 tolerance.

```toml
anr = { version = "0.1.1", default-features = false, features = ["openvino"] }
```

```rust,no_run
use anr::{openvino::{DFN3, Device}, NoiseReduction, Result};

fn main() -> Result<()> {
    let mut denoiser = DFN3::from_file_with_device("models/dfn3.onnx", Device::Cpu)?;
    // Device::Gpu(0) and Device::Npu request exactly those plugins.
    println!("execution device: {}", denoiser.execution_device());
    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    denoiser.run(DFN3::SAMPLE_RATE, &input, &mut output)?;
    Ok(())
}
```

`Device::Gpu(n)` selects `GPU.n`; the text parser also accepts `CPU`, `GPU` (= `GPU.0`), and `NPU`. AUTO/HETERO and fallback lists are not supported. Missing runtime/device or model-compilation failures return errors, never switch a requested GPU/NPU to CPU.

#### SDKs and deployment

TensorRT requires a C++ toolchain, TensorRT SDK (`TENSORRT_ROOT`), and CUDA SDK (`CUDA_ROOT` or `CUDA_PATH`). `TENSORRT_LIB_DIR` / `CUDA_LIB_DIR` override link-library directories. OpenVINO requires its C++ Runtime/ONNX frontend SDK through `OPENVINO_ROOT` or `OpenVINO_DIR` (CMake directory). The official wheel's `include/` and `libs/` are accepted; Python is not a product runtime dependency. No system-wide SDK installation is required.

PowerShell examples below use conventional paths: replace them with your extracted SDK locations, then run Cargo from the workspace root:

```powershell
$env:TENSORRT_ROOT = 'C:\SDKs\TensorRT-10.16.1'
$env:CUDA_ROOT = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
$env:OPENVINO_ROOT = 'C:\SDKs\openvino'
cargo build -p anr --release --no-default-features --features cuda,opencl
cargo build -p anr --release --no-default-features --features tensorrt,openvino
```

CUDA needs an NVIDIA driver and discoverable NVRTC/builtins libraries (`PATH` on Windows or `LD_LIBRARY_PATH` on Linux); OpenCL needs a vendor GPU ICD. TensorRT/OpenVINO `build.rs` compile private shared C++ shims, loaded by Rust with `libloading`. Deploy `anr_tensorrt_native.dll` / `anr_openvino_native.dll` (Linux: `libanr_tensorrt_native.so` / `libanr_openvino_native.so`) beside your executable, plus the vendor runtimes, frontend/device plugins, and required notices. A build-tree path is not portable deployment. `ANR_TENSORRT_SHIM` / `ANR_OPENVINO_SHIM` select shim files; on Windows `ANR_TENSORRT_RUNTIME` selects a TensorRT DLL directory or file, while `ANR_OPENVINO_RUNTIME` selects the runtime library file. Standard loader paths must resolve their dependencies. Default application ZIPs remain CPU ORT and do not automatically bundle these SDKs/runtimes.

Native `cuda` requires Driver API 12.0+ and NVRTC major 12 or 13, preferring 13 before 12. It checks required driver symbols and reports unsupported runtimes as ordinary errors. Windows loads `nvrtc64_130_0.dll` / `nvrtc64_120_0.dll`; Linux loads `libnvrtc.so.13` / `.12` / unversioned `libnvrtc.so`. The driver must support the selected NVRTC's PTX, and NVRTC must support the GPU architecture; NVRTC 13 is not guaranteed to work with arbitrary 12.x drivers. NVRTC's builtins dependencies must be on the loader path too.

#### CLI, benchmark, and recorded validation

Facade/CLI features are opt-in and do not change the CLI's ORT default. `--provider` is ORT-only; `--openvino-device CPU|GPU|GPU.n|NPU` is OpenVINO-only. With SDK/runtime paths set:

```text
cargo run -p anr-cli --release --no-default-features --features cuda -- denoise voice.wav --model models/dfn3.onnx --backend cuda --output clean.cuda.wav
cargo run -p anr-cli --release --no-default-features --features openvino -- denoise voice.wav --model models/dfn3.onnx --backend openvino --openvino-device CPU --output clean.openvino.wav
cargo run -p anr-bench --release -- --backend cuda,opencl --warmup 100 --frames 1000 --json
cargo run -p anr-bench --release --features tensorrt,openvino -- --backend gl,vk,dx,cuda,tensorrt,openvino,opencl --openvino-device CPU --warmup 100 --frames 1000 --json
```

`anr-bench` compiles CUDA/OpenCL into its default native build but still selects `gl,vk,dx` by default; TensorRT/OpenVINO require the named SDK features. All selections use the same input, warmup/state continuity, and timed-loop rules. JSON configuration includes `openvino_device`, successful OpenVINO results include `execution_device`, and native `cpu_fallback_allowed` is false **even for explicitly selected OpenVINO CPU**. It describes fallback policy, not GPU-only execution.

Recorded Windows validation passed streaming references for all four backends and 10 real raw/WAV/stereo/batch CLI checks: CUDA/TensorRT/OpenCL on RTX 2080 Ti, OpenVINO CPU on AMD Ryzen 7 5800X; CUDA 13.3, TensorRT 10.16.1, OpenVINO 2026.4.0. All seven native benchmark selections completed 100 warmup and 1,000 measured frames each. OpenVINO GPU failed compiling the original graph's `NonZero_count` kernel; NPU was not tested. Cross-platform GPU and OpenVINO GPU/NPU success are not established. See the [root README](../../README.md#benchmark-native-compute-and-ort-gpu-inference) for full benchmark semantics and the separate ORT-provider path.

### License

The code is **MIT OR Apache-2.0**; choose either. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).

Check model weights and ONNX files separately for use and redistribution rights. The DeepFilterNet source being dual-licensed does not mean every model from every source has the same license. The Rust `ort` crate, ONNX Runtime, and execution-provider libraries also have their own license and distribution requirements.

## 简体中文

`anr` 提供 DFN3 的 Rust 接口，后端包括 ORT、OpenGL、GLES、Vulkan、wgpu、D3D12、CUDA、TensorRT、OpenVINO 和 OpenCL。它处理 **48 kHz、单声道 `f32` 采样**，输入输出等长。默认不启用任何后端。启用 `ort` 后可使用 `anr::DFN3`；启用其他后端后，从对应模块使用，例如 `anr::cuda::DFN3`。

它不读取麦克风、不解码 WAV，也不转换采样率。需要 **最新 Rust stable 工具链**；默认 features 会在构建时下载并准备受支持的 DFN3 模型。

### 选择后端

| Cargo 中填写的名称 | 后端 | 对应类型 |
| --- | --- | --- |
| `ort` | ONNX Runtime | `anr::DFN3` |
| `gl` | OpenGL | `anr::gl::DFN3` |
| `gles` | OpenGL ES | `anr::gles::DFN3` |
| `vk` | Vulkan | `anr::vk::DFN3` |
| `wgpu` | wgpu | `anr::wgpu::DFN3` |
| `dx` | Direct3D 12 | `anr::dx::DFN3` |

只用 ORT 以外的后端时，在 Cargo 配置中设置 `default-features = false`，并在 `features` 中填写后端名称。例如：

```toml
[dependencies]
anr = { version = "0.1.1", default-features = false, features = ["wgpu"] }
```

在 `features` 中填写要用的后端。写多个就会同时编入多个，程序需要自行决定使用哪一个；它不会自动切换。非 ORT 后端只支持本项目指定的 DFN3 模型，不能拿来运行任意 ONNX 文件。硬件和系统要求见下文。原生 `cuda` / `tensorrt` 是独立实现，与 ORT 的 `ort-cuda` / `ort-tensorrt` provider 不同。

### ORT 入门

```toml
[dependencies]
anr = { version = "0.1.1", features = ["ort"] }
```

依赖示例使用 `0.1.1`。使用本地源码时，可以把版本依赖改成指向对应 crate 目录的 `path` 依赖。
| `cuda` | 原生 CUDA Driver/NVRTC | `anr::cuda::DFN3` |
| `tensorrt` | 原生 TensorRT + CUDA DSP/FFT | `anr::tensorrt::DFN3` |
| `openvino` | 原生 OpenVINO Runtime，明确选择 CPU/GPU/NPU | `anr::openvino::DFN3`、`anr::openvino::Device` |
| `opencl` | 原生 OpenCL 1.2 GPU | `anr::opencl::DFN3` |

```rust,no_run
use anr::{DFN3, NoiseReduction, Result};

fn main() -> Result<()> {
    anr::ort::initialize_runtime()?;
    let environment = anr::ort::init().build().map_err(anr::Error::backend)?;
    let mut denoiser = DFN3::from_file(&environment, anr::default_model_path()?)?;
    denoiser.set_factor(80.0);

    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    denoiser.run(DFN3::SAMPLE_RATE, &input, &mut output)?;
    denoiser.reset();
    Ok(())
}
```

示例输入是静音，使用时换成自己的音频采样。`default_model_path()` 先查找可执行文件旁的 `models/dfn3.onnx`，再使用构建时准备的模型，不依赖工作目录，也不会在运行时下载。

### 模型下载与复制

`download-models` 和 `copy-models` 均默认启用，但不会启用任何推理后端。下载器从 [固定的 DFN3 Release](https://github.com/lona-cn/anr/releases/download/model-dfn3-fe5eb64fa2e4/dfn3.onnx) 获取模型，校验 SHA-256 `fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9`，并复用 `$CARGO_HOME/anr/models/dfn3-fe5eb64fa2e4/dfn3.onnx`；未设置 `CARGO_HOME` 时使用 `~/.cargo`。

`copy-models` 将选中的文件复制到 Cargo profile 目录及其 `examples`、`deps` 目录中的 `models/dfn3.onnx`，支持自定义 target 目录和交叉编译目录。它不会自动启用 `download-models`。**构建前将 `ANR_MODEL_PATH` 设为本地文件的绝对路径**，即可跳过下载，或单独使用 `copy-models`；源文件变化时会重新复制。

仅启用 `download-models` 时，`default_model_path()` 使用缓存文件，不复制模型。`cargo install` 的临时构建目录删除后，也可使用这份持久缓存。向其他机器分发程序时，须把 `models/dfn3.onnx` 放在可执行文件旁，并附上模型许可及通知，不能依赖构建机器的缓存路径。默认下载器与 ORT 下载器一样使用 native TLS。

设置 `default-features = false` 可关闭自动模型准备及下载器依赖，仍可显式指定模型路径，或使用程序旁的模型。缓存缺失且设置了 `CARGO_NET_OFFLINE=true`、下载失败或摘要不符时，构建会报错；离线构建可设置 `ANR_MODEL_PATH`。docs.rs 文档构建跳过模型准备。

### 每路连续音频各用一个降噪器

连续录音要一直使用同一个 `DFN3` 降噪器，不要让几路无关音频轮流共用。模型处理后面的音频时会用到前面的信息。换录音或重新从头播放前调用 `reset()`；它会清除音频状态，保留降噪强度。

每次尽量传入 480 的整数倍。`DFN3::FRAME_SIZE = 480`，也就是 10 ms 音频。`run` 接受任意非零长度，但末尾不足 480 个采样时会补零，按完整的一帧处理，再只返回原有长度的结果。补上的零也会影响后续处理，所以拆成许多短段，结果可能与整段处理不同。

库不会消除模型带来的时间延迟，也没有在录音结束后额外取出剩余结果的操作。多声道音频需要由应用先分开处理。

### 调用前先检查输入和输出

- 输入至少有一个采样。
- 采样率必须是 48,000 Hz。
- 输出数组由你分配，长度必须与输入相同。

这三项检查失败时，输出和处理状态都不会改变。如果运行模型时才出错，输出和状态可能已经部分更新，不会撤销已经完成的处理。

未启用后端 feature 时，`anr` 导出 `NoiseReduction`、`Error`、`Result` 和 `default_model_path`；默认的 `ort-download-binaries` 不会单独启用 ORT。`NoiseReduction` 支持 `Box<dyn NoiseReduction>`；`Send` 允许跨线程移交实例，不保证并行调用或非阻塞执行。

### 调整降噪强度

所有 DFN3 实现都通过 `NoiseReduction` 提供 `set_factor(0..=100)`，按 `round(factor / 100 * 60)` 换算成分贝。只有 `anr::ort::DFN3` 另外提供 `set_atten_lim_db` / `get_atten_lim_db`，用于直接访问 `0..=60` dB；它们不属于通用 trait 或 Compute API。

超出范围的值会被限制到最近的上限或下限；`NaN` 不改变当前设置，正无穷按上限处理，负无穷按下限处理。

`set_factor` 会将分贝值取整，ORT 专属的分贝 setter 保留小数。`factor()` 根据实际设置换算并取整，可能与传入值略有不同。强度是衰减上限，不是原音与处理结果的混合比例；零仍运行所提供的模型，不表示原音旁路。

### 用 ORT raw 音频示例试运行

在项目根目录执行：

```text
cargo run -p anr --features ort --example denoise -- <input.f32le> <output.f32le>
```

输入文件是 **48 kHz、单声道的原始采样，不是 WAV**：每个采样占 4 字节，以小端序保存为 IEEE 754 `f32`。输出格式相同。这个命令需要本项目源码，只安装 Rust 包不会得到整个项目。

需要其他模型时，在输入、输出参数前加上模型路径。

### ORT 模型要求

ORT 的 `DFN3::from_file(&environment, path)` 使用应用持有的 `Environment` 加载模型，`DFN3::new(session)` 使用已配置的 session。两种方式都只接受下表的三个输入和三个输出，类型为 `f32`，输入排列不限。`[]` 是单个数，`[1]` 是单元素的一维数组。非 ORT 的 `from_file` 构造器只传 `path`，并要求后文指定的固定模型摘要；OpenVINO 另提供 `from_file_with_device`。

| 方向 | 名称 | 运行时的大小 |
| --- | --- | --- |
| 输入 | `input_frame` | `[480]` |
| 输入 | `states` | `[45304]` |
| 输入 | `atten_lim_db` | `[]` 或 `[1]` |
| 输出 | `enhanced_audio_frame` | `[480]` |
| 输出 | `new_states` | `[45304]` |
| 输出 | `lsnr` | `[]` 或 `[1]` |

模型可以给一维数组的长度取名，而不写固定数值。加载时会检查维数和已经写明的长度；符号长度不能改变运行时需要的 480 个采样和 45,304 个状态值。对应常量是 `DFN3::SAMPLE_RATE = 48000`、`FRAME_SIZE = 480`、`STATE_SIZE = 45304`。

音频和状态输出会写入事先分配的空间，ONNX Runtime 会检查实际长度。`lsnr` 只检查模型声明的类型和大小，不读取或返回计算结果。同样叫 DeepFilterNet3 的模型也可能使用不同的输入输出，不能只看名字判断是否可用。

### 选择 features 和 ORT 运行库

- `ort`：提供 `DFN3`；保留默认 features 时，在构建中下载并复制 CPU 运行库，需要网络和平台支持。
- `gl`、`gles`、`vk`、`wgpu`、`dx`：启用对应 Compute 后端和 `anr::<backend>::DFN3`，不会启用 ORT。
- `cuda`、`tensorrt`、`openvino`、`opencl`：启用独立原生后端和对应 `anr::<backend>::DFN3` 模块，不启用 ORT。TensorRT/OpenVINO 构建需要 C++ SDK，因此启用全部 features 也需要这些 SDK。
- `ort-download-binaries`：默认启用的弱运行库下载选项，不会单独激活 `anr-ort`；自动模型准备与 ORT 无关。
- `download-models`：默认启用，下载并缓存固定 DFN3 模型；设置 `ANR_MODEL_PATH` 时改用本地文件。
- `copy-models`：默认启用，将选中的模型复制到构建产物旁，与下载独立控制。
- `ort-load-dynamic`：加载自备运行库。关闭 defaults 可去掉下载器和 TLS 依赖；初始化前将 `ORT_DYLIB_PATH` 设为兼容的共享库文件。
- `ort-cuda`、`ort-tensorrt`、`ort-directml`、`ort-coreml`、`ort-xnnpack`：编入对应库的支持，不会自动选择 GPU，也不会安装驱动或其他必需的库。

当前固定使用 `ort = 2.0.0-rc.15` 和 API 27，不启用 `ndarray`。默认预编译运行库为 1.31.0，也可用兼容 API 27 的库，例如 1.28.0。`anr::ort` 提供同一版本的运行库类型。

```toml
anr = { version = "0.1.1", default-features = false, features = ["ort-load-dynamic"] }
```

在 `anr::ort::init().build()` 前调用 `anr::ort::initialize_runtime()`，让动态加载失败正常返回，避免进入上游的隐式加载路径。这个函数不创建环境，环境由应用持有和配置，多个降噪实例共享同一个 `Environment`。`DFN3::from_file(&environment, path)` 显式选择内置 CPU provider，以单线程执行算子并禁用忙等待。若要选其他 provider，先配置 `SessionBuilder`，调用 `anr::ort::session_from_file(builder, path, &providers)`，再传给 `DFN3::new(session)`。

ORT GPU Session 可以让不支持或更适合 CPU 的算子使用 CPU kernel。库保留调用方 Session 的算子分配，CLI 和 benchmark 也默认允许这种混合执行。显式 provider 注册失败仍报错，不会把不可用的 GPU provider 隐式换成 CPU。

共用加载入口在创建 CUDA/TensorRT Session 前于内存中转换图，不覆盖原 ONNX，也不在每帧重复转换。CUDA 展开不支持的 sigmoid 卷积融合；TensorRT 另转换符合条件的静态状态序列和可安全展开的张量分支。CPU、DirectML 与原生 Compute 保持原样。已有模型字节的调用方可使用 `anr::ort::prepare_model` 和 `ModelTarget`；直接用上游 API 创建 Session 会绕过这一步。


### 不使用 ORT 的 Compute 后端

例如启用 `default-features = false, features = ["gles"]`，使用 `anr::gles::DFN3::from_file(path)` 创建实例；其余后端对应 `gl`、`vk`、`wgpu`、`dx` feature 和模块。均实现 `NoiseReduction`，每实例持有工作线程、GPU 常驻权重和状态，`run` 同步返回。`anr-dfn3` 共用固定执行顺序和 WGSL 算子，原生 GL/GLES/Vulkan/D3D12 使用 Naga 生成 GLSL/GLSL ES/SPIR-V/HLSL，不替换成 ORT 或 CPU 推理。

```toml
anr = { version = "0.1.1", default-features = false, features = ["gles"] }
```

```rust,no_run
use anr::{gles::DFN3, NoiseReduction, Result};

fn main() -> Result<()> {
    let mut denoiser = DFN3::from_file("models/dfn3.onnx")?;
    denoiser.set_factor(80.0);
    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    denoiser.run(DFN3::SAMPLE_RATE, &input, &mut output)?;
    denoiser.reset();
    Ok(())
}
```

`gles` 需要 Windows/WGL 或 Linux 硬件 EGL/pbuffer 创建真实 ES 3.1+ context，复用 `anr-gl` 执行器，但不退回桌面 GL。高级 GL 调用方可用 `DFN3::from_file_with_api(path, api)` 显式选择 `GlApi`；普通 `anr::gl::DFN3::from_file` 仍只创建桌面上下文。GLES 后端未提供 Android、WebGL 或 macOS context。

`gl` 需 Windows/Linux 桌面 GL 4.3；`vk` 需 Windows/Linux 硬件 Vulkan 1.1；`wgpu` 在 Windows/Linux/macOS 选择原生硬件 Vulkan/DX12/Metal；`dx` 需 Windows 硬件 D3D12 与系统 D3DCompiler 47，不使用 WARP、DirectML 或 wgpu。五种均已在 Windows / RTX 2080 Ti 实跑，其他平台的 GPU 路径未验证。

对于上述 GL/GLES/Vulkan/wgpu/D3D12 shader 后端，原始外置 ONNX 只提供参数，要求 SHA-256 为 `fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9`。无需模型转换、Python 或外部 shader 编译器。其他模型或不支持的驱动明确报错。强度为零保留原模型的衰减分支，不表示原音旁路。

### 原生 CUDA、TensorRT、OpenVINO 和 OpenCL

这些后端校验同一固定模型摘要，实现 `NoiseReduction`，每路流持有一个工作线程，`run` 同步等待结果。CUDA 与 OpenCL 使用固定 FP32 调度执行包括 FFT、归一化、循环网络、频谱滤波和 overlap-add 的完整流程，权重与状态常驻 GPU。CUDA 使用 Driver API、NVRTC 编译和 replay graph；OpenCL 需要支持 1.2 的硬件 GPU ICD，不会退回 CPU device。两者均不要求构建时 SDK。

TensorRT 使用两个真正的标准层神经网络 engine 加上 CUDA DSP/FFT graph，不是 ORT、ONNX parser 或单体推理 plugin。采用强类型 FP32，禁用 TF32；初始化时构建 engines，没有持久 engine cache。已记录的一次初始化耗时为 17.342 秒，不代表快速冷启动。

OpenVINO 使用原生 frontend 直接导入原 ONNX，而不是只从 ONNX 读取参数。它使用 `MakeStateful`（`ReadValue`/`Assign`），每个降噪器独占一个 `InferRequest`，状态保留在所选 plugin 内，不逐帧往返主机。`from_file` 默认**明确选择 CPU**，不是 GPU fallback。CPU/GPU 请求偏重精度的 FP32 推理；NPU 硬件可能使用 FP16，尚未验证满足 FP32 reference 阈值。

```toml
anr = { version = "0.1.1", default-features = false, features = ["openvino"] }
```

```rust,no_run
use anr::{openvino::{DFN3, Device}, NoiseReduction, Result};

fn main() -> Result<()> {
    let mut denoiser = DFN3::from_file_with_device("models/dfn3.onnx", Device::Cpu)?;
    // Device::Gpu(0) 和 Device::Npu 只请求对应 plugin。
    println!("execution device: {}", denoiser.execution_device());
    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    denoiser.run(DFN3::SAMPLE_RATE, &input, &mut output)?;
    Ok(())
}
```

`Device::Gpu(n)` 选择 `GPU.n`；文本解析也接受 `CPU`、`GPU`（即 `GPU.0`）、`NPU`，不支持 AUTO/HETERO 或 fallback 列表。运行库或设备缺失、模型编译失败均报错，不会将请求的 GPU/NPU 换成 CPU。

#### SDK 与部署

TensorRT 需要 C++ 工具链、TensorRT SDK（`TENSORRT_ROOT`）及 CUDA SDK（`CUDA_ROOT` 或 `CUDA_PATH`）；可用 `TENSORRT_LIB_DIR` / `CUDA_LIB_DIR` 覆盖链接库目录。OpenVINO 需要包含 Runtime/ONNX frontend 的 C++ SDK，通过 `OPENVINO_ROOT` 或指向 CMake 目录的 `OpenVINO_DIR` 配置。官方 wheel 的 `include/`、`libs/` 布局也可使用；Python 不是产品运行依赖，无需系统级安装 SDK。

以下 PowerShell 示例使用常规路径，请替换为自己的 SDK 解压目录，再从工作区根目录执行 Cargo：

```powershell
$env:TENSORRT_ROOT = 'C:\SDKs\TensorRT-10.16.1'
$env:CUDA_ROOT = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
$env:OPENVINO_ROOT = 'C:\SDKs\openvino'
cargo build -p anr --release --no-default-features --features cuda,opencl
cargo build -p anr --release --no-default-features --features tensorrt,openvino
```

CUDA 需要 NVIDIA 驱动，并让加载器找到 NVRTC/builtins 库（Windows `PATH`、Linux `LD_LIBRARY_PATH`）；OpenCL 需要厂商 GPU ICD。TensorRT/OpenVINO 的 `build.rs` 编译私有共享 C++ shim，由 Rust 通过 `libloading` 加载。分发时将 `anr_tensorrt_native.dll` / `anr_openvino_native.dll`（Linux 为 `libanr_tensorrt_native.so` / `libanr_openvino_native.so`）放在可执行文件旁，并携带厂商 runtime、frontend/设备 plugin 及必要许可说明；构建目录路径不能替代可移植部署。`ANR_TENSORRT_SHIM` / `ANR_OPENVINO_SHIM` 指向 shim 文件；Windows 上 `ANR_TENSORRT_RUNTIME` 可指向 TensorRT DLL 目录或文件，`ANR_OPENVINO_RUNTIME` 则指向 runtime 库文件。标准加载路径也必须能解析依赖。默认应用 ZIP 仍是 CPU ORT，不会自动打包这些 SDK/运行库。

原生 `cuda` 要求 Driver API 12.0+ 和 major 12 或 13 的 NVRTC，优先加载 13、其次 12。它预检所需 driver 符号，不支持的运行库正常返回错误。Windows 加载 `nvrtc64_130_0.dll` / `nvrtc64_120_0.dll`，Linux 加载 `libnvrtc.so.13` / `.12` / 无版本号的 `libnvrtc.so`。驱动必须支持所选 NVRTC 的 PTX，NVRTC 也须支持 GPU 架构；不保证 NVRTC 13 可搭配任意 12.x 驱动。NVRTC 的 builtins 依赖同样须位于加载路径。

#### CLI、benchmark 和实测范围

facade/CLI 的新后端均显式选择，不改变 CLI 默认 ORT。`--provider` 仅用于 ORT；`--openvino-device CPU|GPU|GPU.n|NPU` 仅用于 OpenVINO。配置好 SDK/运行库路径后：

```text
cargo run -p anr-cli --release --no-default-features --features cuda -- denoise voice.wav --model models/dfn3.onnx --backend cuda --output clean.cuda.wav
cargo run -p anr-cli --release --no-default-features --features openvino -- denoise voice.wav --model models/dfn3.onnx --backend openvino --openvino-device CPU --output clean.openvino.wav
cargo run -p anr-bench --release -- --backend cuda,opencl --warmup 100 --frames 1000 --json
cargo run -p anr-bench --release --features tensorrt,openvino -- --backend gl,vk,dx,cuda,tensorrt,openvino,opencl --openvino-device CPU --warmup 100 --frames 1000 --json
```

`anr-bench` 默认原生构建已编入 CUDA/OpenCL，但默认选择仍为 `gl,vk,dx`；TensorRT/OpenVINO 需要同名 SDK features。各后端使用相同输入、预热/状态连续性及计时循环规则。JSON 配置含 `openvino_device`，OpenVINO 成功结果含 `execution_device`；原生后端的 `cpu_fallback_allowed` 均为 false，**明确选择 OpenVINO CPU 也如此**。它表示 fallback 策略，不代表仅在 GPU 执行。

已有 Windows 实测：四后端连续帧 reference 和共 10 项真实 raw/WAV、立体声、批处理 CLI 检查通过；CUDA/TensorRT/OpenCL 在 RTX 2080 Ti，OpenVINO CPU 在 AMD Ryzen 7 5800X，SDK 为 CUDA 13.3、TensorRT 10.16.1、OpenVINO 2026.4.0。七个原生 benchmark 后端均完成各 100 帧预热、1000 帧正式测量。OpenVINO GPU 编译因原图 `NonZero_count` kernel 选择失败；NPU 未测。不能据此声称跨平台 GPU 或 OpenVINO GPU/NPU 验证通过。完整 benchmark 规则及独立的 ORT provider 路径见[根 README](../../README.md#测试原生-compute-和-ort-gpu-推理性能)。

### 许可

代码采用 **MIT OR Apache-2.0**，使用时任选其一。完整文本见 [LICENSE-MIT](LICENSE-MIT) 和 [LICENSE-APACHE](LICENSE-APACHE)。

模型权重和 ONNX 文件需要单独确认使用与分发权限。DeepFilterNet 源码采用双许可，不代表任何来源的模型都具有相同许可。Rust `ort` 包、ONNX Runtime 及执行 provider 所依赖的库，也各有自己的许可和分发要求。
