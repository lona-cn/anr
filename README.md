# anr

[English](#english) | [简体中文](#简体中文)

## English

`anr` is a Rust audio denoising library and command-line tool. It runs DeepFilterNet3 (DFN3) with ONNX Runtime, or model-specific OpenGL, OpenGL ES, Vulkan, wgpu, and Direct3D 12 Compute backends without ORT.

The library takes **48 kHz, mono `f32` samples**. For audio files, resampling, and multiple channels, use [`anr-cli`](crates/anr-cli/README.md). The CLI handles files, batches, and raw PCM streams; it does not provide microphone access, virtual audio devices, or drivers.

Source builds require the **latest stable Rust toolchain**. Rust packages and `cargo install` do not include or automatically download a model. GitHub application ZIPs use the matched DFN3 model described below; a model with the same name is not necessarily compatible, so check its inputs and outputs before substituting one.

### Choose a backend for the Rust library

The library supports `ort` (ONNX Runtime), `gl` (OpenGL), `gles` (OpenGL ES), `vk` (Vulkan), `wgpu`, and `dx` (Direct3D 12). No library backend is enabled by default; the default `ort-download-binaries` option does not turn on ORT.

For example, build with `wgpu` and without ORT:

```toml
[dependencies]
anr = { version = "0.1.1", default-features = false, features = ["wgpu"] }
```

In `Cargo.toml`, list the backends your program needs under `features`. If you list more than one, Cargo includes them all; it will not switch between them while the program runs. The five non-ORT backends only support the project's specified DFN3 model, not arbitrary ONNX files. See the [library README](crates/anr/README.md) for backend requirements.

### Denoise a file

The [GitHub application releases](https://github.com/lona-cn/anr/releases) are built as native ZIPs with the CLI, CPU runtime, `models/dfn3.onnx`, and license files. Extract the whole ZIP and run from its directory:

```sh
./anr denoise voice.wav --output clean.wav --model models/dfn3.onnx
```

On Windows, use `.\anr.exe` instead of `./anr`. The `--model` argument is still required. Application ZIPs appear only after the release workflow passes; the separate [model asset release](https://github.com/lona-cn/anr/releases/tag/model-dfn3-fe5eb64fa2e4) contains the model, not a CLI build. See the [CLI distribution notes](crates/anr-cli/README.md) for system runtime requirements.

Install the CLI from the workspace root. The executable is called `anr`. This default build uses ORT's prebuilt libraries where available; on Intel macOS, follow the [CLI instructions](crates/anr-cli/README.md) for supplying your own runtime instead.

```text
cargo install --path crates/anr-cli --locked
anr denoise voice.wav --output clean.wav --model /path/to/dfn3.onnx
anr denoise a.flac b.mp3 --output-dir cleaned --model /path/to/dfn3.onnx
anr inspect voice.wav --json
anr backends --json
```

Replace the model path with your own. Once installed, `anr` does not need to run from the source directory. It uses CPU ORT by default, preserves the input sample rate, channel count, and frame count unless you request otherwise, and writes WAV or raw f32le.

See the [CLI README](crates/anr-cli/README.md) for raw stdin/stdout, how output files are saved, execution providers, model latency, and binary-safe shell examples.

The default build may download ONNX Runtime, so it needs network access and a supported platform toolchain. Intel macOS is not covered by the current default ORT prebuilt distribution. CI is configured to use Rust stable on native Windows x64, Linux x64, and macOS arm64/x64; the configuration itself is not evidence that those jobs have passed. See the CLI README for recorded local results and platforms not yet tested.

For a source build without ONNX Runtime, run the GLES backend directly from the workspace:

```text
cargo run -p anr-cli --release --no-default-features --features gles -- denoise voice.wav --model models/dfn3.onnx --backend gles --output clean.gles.wav
```

This requires a supported native ES 3.1+ driver and the exact model described below. `--backend gles` is required even in a GLES-only build: the CLI's default selection remains `ort`.

### Use the Rust library

The `anr` library does not enable a backend by default. Choose `ort`, `gl`, `gles`, `vk`, `wgpu`, or `dx`; the following example explicitly enables ORT and its `DFN3` reexport:

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
    let mut denoiser = DFN3::from_file(&environment, "models/dfn3.onnx")?;
    denoiser.set_factor(80.0);

    // Replace this silence with your own 48 kHz mono samples.
    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    denoiser.run(DFN3::SAMPLE_RATE, &input, &mut output)?;

    // Clear the previous recording's state; keep the strength setting.
    denoiser.reset();
    Ok(())
}
```

Before calling `run`:

- Supply at least one input sample.
- Use a sample rate of 48,000 Hz.
- Allocate an output slice with the same length as the input.

If any of these checks fail, the output and processing state stay unchanged. An error during model execution may leave both partly updated; it does not undo work already done.

#### Keep continuous audio in chunks of 480 samples

The model processes **480 samples at a time: 10 ms of audio**. `run` accepts any nonzero length, but pads a short final frame with zeros, processes the whole frame, and returns only the requested samples. That padding also advances the model state. Splitting a recording into many short calls can therefore produce different results from processing it in one call. For live audio, collect samples into multiples of 480 before passing them in.

Keep one `DFN3` instance per continuous stream; don't share it between unrelated recordings. The model uses information from earlier audio. Call `reset()` before switching recordings or restarting playback. This clears the audio state without changing the strength.

The library does not compensate for model latency and has no end-of-stream flush operation. Sound quality, latency, and suitability depend on the actual model; matching the required inputs and outputs does not establish any of them.

#### Adjust the strength

Every DFN3 backend supports `NoiseReduction::set_factor(0..=100)`, mapped to dB with `round(factor / 100 * 60)`. Only the ORT concrete type additionally exposes `set_atten_lim_db` / `get_atten_lim_db` for direct `0..=60` dB access; these methods are not on the shared trait or compute frontends.

Out-of-range values are clamped. `NaN` leaves the setting unchanged; positive and negative infinity use the upper and lower bounds respectively.

`set_factor` rounds to whole dB; ORT's direct dB setter preserves fractional values. `factor()` converts the actual dB setting back and rounds it, so it may differ from the value passed in. This is an attenuation limit, not a dry/wet mix. Zero still runs the model and is not a bypass for the supplied DFN3 model.

#### Try the raw audio example

The following example uses ORT; run it from the workspace root. For compute backends and audio containers, use the CLI with an explicit `--backend` instead.

```text
cargo run -p anr --features ort --example denoise -- <model.onnx> <input.f32le> <output.f32le>
```

The input is **48 kHz mono raw audio, not WAV**: four bytes per sample, stored as little-endian IEEE 754 `f32`. The output uses the same format. In this checkout, `<model.onnx>` can be `models/dfn3.onnx`; that file is not included in the Rust packages. Check that the model meets the requirements below and that you have permission to use it.

### ORT model inputs and outputs

The ORT adapter requires exactly these three inputs and three outputs, all `f32`. Names are matched when loading; input order does not matter. Brackets show tensor shapes: `[]` is a scalar, and `[1]` is a one-element vector. Compute backends require the fixed SHA-256 model in the next section, not merely matching port shapes.

| Direction | Name | Runtime shape |
| --- | --- | --- |
| Input | `input_frame` | `[480]` |
| Input | `states` | `[45304]` |
| Input | `atten_lim_db` | `[]` or `[1]` |
| Output | `enhanced_audio_frame` | `[480]` |
| Output | `new_states` | `[45304]` |
| Output | `lsnr` | `[]` or `[1]` |

A one-dimensional tensor may declare a symbolic length. Loading checks the rank and any fixed lengths; symbolic lengths do not change the required 480 audio samples and 45,304 state values at runtime. The constants are `DFN3::SAMPLE_RATE = 48000`, `FRAME_SIZE = 480`, and `STATE_SIZE = 45304`.

Audio and state outputs use preallocated buffers, and ONNX Runtime checks their actual lengths. For `lsnr`, only the declared type and shape are checked; its result is neither read nor returned. Extra inputs or outputs, different ranks, or a different meaning or ordering of state values require changes to the implementation.

### Select a backend and runtime

Adding `anr` alone does not enable model execution. Choose the Cargo features you need:

| Feature | What it does |
| --- | --- |
| `ort` | Provides `DFN3`. With default features enabled, the CPU runtime is downloaded and copied at build time. |
| `gl` | Provides `anr::gl::DFN3`, using native OpenGL 4.3 compute shaders without ORT. |
| `gles` | Provides `anr::gles::DFN3`, using a real native OpenGL ES 3.1+ compute context without ORT. |
| `vk` | Provides `anr::vk::DFN3`, using native Vulkan 1.1 compute without ORT or wgpu. |
| `wgpu` | Provides `anr::wgpu::DFN3`, using wgpu hardware adapters without ORT. |
| `dx` | Provides `anr::dx::DFN3`, using native Windows Direct3D 12 compute, not DirectML or wgpu. |
| `ort-download-binaries` | Default download option; takes effect only when `ort` is enabled and does not enable a backend by itself. |
| `ort-load-dynamic` | Loads your runtime. Use `default-features = false` with this feature to omit the downloader and its TLS dependencies; set `ORT_DYLIB_PATH` to the shared library file. |
| `ort-cuda` / `ort-tensorrt` | Compiles support for CUDA / TensorRT. |
| `ort-directml` / `ort-coreml` / `ort-xnnpack` | Compiles support for DirectML / CoreML / XNNPACK. |

The project pins `ort = 2.0.0-rc.15` with API 27, without `ndarray`. Its default prebuilt runtime is 1.31.0; compatible self-provided API-27 runtimes, including 1.28.0, can also be used. Set `ORT_DYLIB_PATH` before initialization; it points to a library file, not a directory.

Call `anr::ort::initialize_runtime()` before building an environment so dynamic-library errors are returned normally. It loads the library but does not create or configure an environment. Build one `Environment` in your application and share it between denoisers. `DFN3::from_file(&environment, path)` explicitly uses the built-in CPU provider, one intra-op thread, and no thread-pool spinning. GPU features do not select a GPU or install drivers. For a custom provider, use `anr::ort::session::Session::builder(&environment)` and pass the resulting session to `DFN3::new(session)`.

### Compute backends without ONNX Runtime

Build the CLI with `cargo build -p anr-cli --release --no-default-features --features gl,gles,vk,wgpu,dx`, then select `--backend gl`, `gles`, `vk`, `wgpu`, or `dx`. You can enable any subset. Do not pass an ORT `--provider`. The default CLI build, backend, and application release ZIPs remain ORT.

For example, a GLES-only library dependency:

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

Construct `anr::gles::DFN3::from_file("models/dfn3.onnx")`, or the matching `gl`, `vk`, `wgpu`, or `dx` module, and use the same `NoiseReduction` trait. No ORT environment, runtime library, Python, model conversion, or external shader compiler is needed. The external ONNX file supplies parameters only, after SHA-256 validation against `fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9`; other exports are rejected.

`anr-dfn3` shares the fixed execution plan, weight layout, worker, and WGSL operators. Naga translates to desktop GLSL 4.3, GLSL ES 310, SPIR-V, or HLSL; wgpu consumes WGSL directly. GLES reuses the executor in `anr-gl`, but requires an embedded context rather than accepting desktop ES compatibility. It maps only the completed output frame after a GPU fence because GLES has no `glGetBufferSubData`. The FFT needs 15,360 bytes of workgroup memory, within the ES 3.1 minimum of 16 KiB.

All inference, including FFT, feature normalization, convolutions, grouped linear layers, five GRUs, spectral filtering, and overlap-add, runs in FP32 compute shaders. Weights and two state buffers remain on GPU. Each reducer owns a worker thread; `run` waits for GPU output and is not a nonblocking audio-callback API. `lsnr` is not exposed, model latency is not compensated, and strength zero retains the attenuation branch rather than bypassing the model.

| Backend | Driver/platform requirement |
| --- | --- |
| `gl` | Windows desktop OpenGL 4.3; Linux hardware EGL device with desktop GL 4.3 and pbuffer, without X11 `DISPLAY`. Not macOS. |
| `gles` | Windows WGL driver with native ES context support; Linux hardware EGL device/pbuffer. Both require OpenGL ES 3.1+. No Android, WebGL, or macOS context is provided. |
| `vk` | Windows or Linux hardware Vulkan 1.1 device with compute queue. No window or surface. |
| `wgpu` | Windows, Linux, or macOS hardware Vulkan, DX12, or Metal adapter, as available on that platform. No browser or software fallback. |
| `dx` | Windows hardware D3D12 device and system D3DCompiler 47. No WARP, DirectML, or separately installed DXC. |

Initialization failures return errors rather than falling back to CPU. All five backends passed streaming reference and raw/WAV/stereo/batch CLI tests on native Windows / RTX 2080 Ti. GLES-only compilation, non-GPU tests, and Clippy also passed in WSL Ubuntu 24.04. These results do not establish cross-device performance or sound quality; Linux GPU paths, Android, and macOS remain unverified.

Set `ANR_DFN3_MODEL` to the absolute model path and run `cargo test -p anr-gl -p anr-gles -p anr-vk -p anr-wgpu -p anr-dx -- --include-ignored --nocapture` on hardware supporting the selected packages. The reference covers strength changes, a partial final frame, reset, thread moves, and invalid-input state preservation. `tools/generate_gl_reference.py` regenerates all five fixtures using development-only `onnx==1.19.1` and `numpy==2.2.6`, not ONNX Runtime.

### Which crate should I use?

Use `anr` in most Rust applications, or install `anr-cli` to work with files. Backend crates separate each implementation from the common interface.

| Crate | When to use it |
| --- | --- |
| `anr` | The usual starting point. Exports `NoiseReduction`, `Error`, and `Result`; enables the corresponding `DFN3` through `ort`, `gl`, `gles`, `vk`, `wgpu`, or `dx`. |
| `anr-core` | You need only `NoiseReduction`, `Error`, and `Result`, without a model runtime dependency. |
| `anr-ort` | You want the DFN3 ONNX Runtime implementation directly, or access to the same version of `ort`. |
| `anr-gl` | You want the fixed-model native OpenGL 4.3 DFN3 implementation without ORT. |
| `anr-gles` | Native OpenGL ES 3.1 DFN3, sharing the GL-family executor without desktop-GL or CPU fallback. |
| `anr-vk` | Native Vulkan compute DFN3, without ORT or wgpu. |
| `anr-wgpu` | wgpu compute DFN3, without ORT. |
| `anr-dx` | Native Windows Direct3D 12 compute DFN3, without ORT, DirectML, or wgpu. |
| `anr-dfn3` | Shared fixed-model parameters, execution plan, worker, and shader sources; not a standalone device backend. |
| `anr-cli` | You want the `anr` executable for files, batches, inspection, and raw PCM streams. CPU ORT is enabled by default. |

`anr-dfn3` and the five compute crates use this root README in their source packages. `anr-core`, `anr-ort`, the facade, and the CLI each have their own README. The compute constructor takes only a model path; an ORT environment or `Session` is never passed to it.

`NoiseReduction` works as `Box<dyn NoiseReduction>`, and reducers can be moved between threads (`Send`). The library crates do not manage audio devices, decode WAV, resample, or split multiple channels. The CLI handles file decoding, resampling, and channels outside the library. No package downloads models automatically.

### Automated GitHub releases

The [workflow](.github/workflows/ci.yml) tests native Windows x64, Linux x64, and macOS arm64/x64. It downloads the pinned model and licenses, verifies their SHA-256 values, and runs both the control graph and real DFN3 from each extracted ZIP before publishing anything.

| Event | Result after all four platforms pass |
| --- | --- |
| Push to `main` or the current `master` branch | Immutable `main-<12-character SHA>` / `master-<12-character SHA>` prerelease; not marked latest. |
| Push `v<workspace version>` | Version release; a Cargo prerelease suffix remains a GitHub prerelease. |
| Pull request or manual CI | Tests and ZIP artifacts only; no Release. |

Publication uses Actions' built-in **`GITHUB_TOKEN`**, mapped to `GH_TOKEN`, with **Contents: write** granted only to the publication job. No custom repository secret is required; PR and build jobs remain read-only.

GitHub can reject a historical target whose workflow changes require **Workflows: write**, which the built-in token cannot obtain ([API restriction](https://docs.github.com/en/rest/releases/releases#create-a-release)). This accepted boundary produces an explicit failed release; the workflow does not substitute a newer commit, move tags, or overwrite assets.

Commit and push the workflow changes before pushing a version tag. For the current `Cargo.toml` version, the matching tag is `v0.1.1`; a mismatched tag fails rather than releasing mislabeled binaries. This workflow does not publish crates.io packages. Existing public releases and assets are never overwritten on rerun.

### License

The code is **MIT OR Apache-2.0**; choose either. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE). Including both full license texts in each standalone package is this project's distribution choice, not a Cargo requirement.

The supplied [model asset release](https://github.com/lona-cn/anr/releases/tag/model-dfn3-fe5eb64fa2e4) is **MIT OR Apache-2.0**, as authorized by the maintainer for the exact SHA-256 in [its manifest](.github/models/dfn3.json); the ZIP carries the model's licenses and notice. Check other models separately: DeepFilterNet's source license does not establish rights for arbitrary exports or weights.

The CLI's Symphonia decoder uses MPL-2.0; rubato, Rust `ort`, ONNX Runtime, and execution-provider libraries have their own terms. See the [third-party notices](crates/anr-cli/THIRD-PARTY-NOTICES.md) for source, modification, and binary distribution obligations. The project's dual license does not replace those terms.

See [CHANGELOG.md](CHANGELOG.md) for version changes and [CI and Release](.github/workflows/ci.yml) for the maintained release procedure.

## 简体中文

`anr` 提供 Rust 音频降噪库和命令行工具，使用 DeepFilterNet3（DFN3）模型，支持 ONNX Runtime，以及不依赖 ORT 的模型专用 OpenGL、OpenGL ES、Vulkan、wgpu 和 Direct3D 12 Compute 后端。

库接口处理 **48 kHz、单声道的 `f32` 采样**。音频文件解码、采样率转换和多声道处理可以交给 [`anr-cli`](crates/anr-cli/README.md)。CLI 用于文件、批处理和 raw PCM 流，不提供麦克风访问、虚拟声卡或驱动。

从源码构建需要 **最新 Rust stable 工具链**。Rust 包和 `cargo install` 不附带或自动下载模型；GitHub 应用 ZIP 则附带下述已匹配的 DFN3 模型。替换模型前仍需核对输入输出，名称相同不代表接口相同。

### 给 Rust 库选择后端

Rust 库支持 `ort`（ONNX Runtime）、`gl`（OpenGL）、`gles`（OpenGL ES）、`vk`（Vulkan）、`wgpu` 和 `dx`（Direct3D 12）。库默认不启用后端；默认的 `ort-download-binaries` 也不会单独启用 ORT。

例如，要使用 `wgpu` 且不启用 ORT，在 `Cargo.toml` 中这样写：

```toml
[dependencies]
anr = { version = "0.1.1", default-features = false, features = ["wgpu"] }
```

在 `Cargo.toml` 的 `features` 中列出要用的后端；写多个就会一起编入程序，程序运行时不会自动切换。五种非 ORT 后端只支持本项目指定的 DFN3 模型，不能用来运行任意 ONNX 文件。各后端的硬件和系统要求见 [anr crate 文档](crates/anr/README.md)。

### 直接处理音频文件

[GitHub 应用 Release](https://github.com/lona-cn/anr/releases) 由 CI 生成原生 ZIP，包含 CLI、CPU 运行库、`models/dfn3.onnx` 和许可文件。完整解压后，进入解压目录运行：

```sh
./anr denoise voice.wav --output clean.wav --model models/dfn3.onnx
```

Windows 将 `./anr` 换成 `.\anr.exe`；`--model` 仍为必填。应用 ZIP 需等发版工作流通过后才会出现；独立的 [模型资产 Release](https://github.com/lona-cn/anr/releases/tag/model-dfn3-fe5eb64fa2e4) 已提供模型，但不是 CLI 安装包。系统运行库要求见 [CLI 文档](crates/anr-cli/README.md)。

在源码工作区根目录安装 CLI，二进制名为 `anr`。默认构建方式使用 ORT 提供的预编译库；Intel macOS 请按 [CLI 文档](crates/anr-cli/README.md) 配置自备运行库。

```text
cargo install --path crates/anr-cli --locked
anr denoise voice.wav --output clean.wav --model /path/to/dfn3.onnx
anr denoise a.flac b.mp3 --output-dir cleaned --model /path/to/dfn3.onnx
anr inspect voice.wav --json
anr backends --json
```

把模型路径换成自己的实际路径。安装后不必从源码目录运行。CLI 默认使用 CPU ORT，默认保留输入采样率、声道数和帧数，输出 WAV 或 raw f32le。

raw stdin/stdout 的格式、输出文件的保存方式、执行 provider、模型延迟及不会损坏二进制音频的 shell 调用示例，见 [CLI 使用文档](crates/anr-cli/README.md)。

默认构建可能下载 ONNX Runtime，需要网络和目标平台工具链。当前 ORT 默认预编译分发不包含 Intel macOS。CI 配置为使用 Rust stable 检查原生 Windows x64、Linux x64、macOS arm64/x64，但配置文件不代表这些任务已经通过；已有本地结果和未实测平台见 CLI 文档。

若要从源码运行不依赖 ONNX Runtime 的 GLES 版本，在工作区根目录执行：

```text
cargo run -p anr-cli --release --no-default-features --features gles -- denoise voice.wav --model models/dfn3.onnx --backend gles --output clean.gles.wav
```

需要受支持的原生 ES 3.1+ 驱动和后文指定的固定模型。即使只编入 GLES，仍须写 `--backend gles`，因为 CLI 的默认选择始终是 `ort`。

### 在 Rust 程序里使用

`anr` 库默认不启用后端，需选择 `ort`、`gl`、`gles`、`vk`、`wgpu` 或 `dx`。以下示例明确启用 ORT 及其 `DFN3` 导出：

```toml
[dependencies]
anr = { version = "0.1.1", features = ["ort"] }
```

依赖示例使用 `0.1.1`。使用本地源码时，可以把版本依赖改成指向对应 crate 目录的 `path` 依赖。

```rust,no_run
use anr::{DFN3, NoiseReduction, Result};

fn main() -> Result<()> {
    anr::ort::initialize_runtime()?;
    let environment = anr::ort::init().build().map_err(anr::Error::backend)?;
    let mut denoiser = DFN3::from_file(&environment, "models/dfn3.onnx")?;
    denoiser.set_factor(80.0);

    // 用实际的 48 kHz 单声道采样替换这段静音。
    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    denoiser.run(DFN3::SAMPLE_RATE, &input, &mut output)?;

    // 清除上一段录音的处理状态，保留降噪强度。
    denoiser.reset();
    Ok(())
}
```

调用 `run` 前：

- 输入至少有一个采样。
- 采样率必须是 48,000 Hz。
- 输出数组由你分配，长度必须与输入相同。

这三项检查失败时，输出和处理状态都不会改变。如果运行模型时才出错，输出和状态可能已经部分更新，不会撤销已经完成的处理。

#### 连续音频每次尽量传入 480 的整数倍

模型一次处理 **480 个采样，也就是 10 ms 音频**。`run` 接受任意非零长度，但末尾不足 480 个采样时会补零，按完整的一帧处理，再只返回原有长度的结果。补上的零也会影响后续处理，所以把录音拆成许多短段，结果可能与整段处理不同。实时收到的零散采样，应先积攒到 480 的整数倍，再交给降噪器。

每路连续音频各用一个 `DFN3` 降噪器，不要让无关录音轮流共用。模型处理后面的音频时会用到前面的信息。换录音或重新从头播放前调用 `reset()`；它会清除音频状态，保留降噪强度。

库不会消除模型带来的时间延迟，也没有在录音结束后额外取出剩余结果的操作。降噪效果、延迟和适用场景取决于实际模型，输入输出匹配并不能证明这些结果。

#### 调整降噪强度

所有 DFN3 后端都支持 `NoiseReduction::set_factor(0..=100)`，按 `round(factor / 100 * 60)` 换算成分贝。只有 ORT 具体类型另外提供 `set_atten_lim_db` / `get_atten_lim_db`，可直接设置和读取 `0..=60` dB；它们不属于通用 trait，Compute 前端也不提供。

超出范围的值会被限制到最近的上限或下限；`NaN` 不改变当前设置，正无穷按上限处理，负无穷按下限处理。

`set_factor` 换算出的分贝值会取整，ORT 的直接分贝 setter 保留小数。`factor()` 根据实际分贝设置换算并取整，可能与传入值略有不同。这是衰减上限，不是原音与处理结果的混合比例。零仍会运行模型，对所提供的 DFN3 模型不表示旁路。

#### 用 raw 音频文件试运行

以下示例使用 ORT，需在工作区根目录运行。Compute 后端和带头音频文件应使用 CLI，并明确指定 `--backend`。

```text
cargo run -p anr --features ort --example denoise -- <model.onnx> <input.f32le> <output.f32le>
```

输入文件是 **48 kHz、单声道的原始采样，不是 WAV**：每个采样占 4 字节，以小端序保存为 IEEE 754 `f32`。输出格式相同。在本项目中，`<model.onnx>` 可以写成 `models/dfn3.onnx`；这个模型文件不随 Rust 包提供。使用前请确认模型符合下面的要求，并且你有权使用它。

### ORT 模型输入和输出

ORT 适配器只接受下表的三个输入和三个输出，类型都必须为 `f32`。加载时按名称匹配，输入排列不限。方括号表示维度和长度；`[]` 是单个数，`[1]` 是单元素的一维数组。Compute 后端另要求下一节指定的模型 SHA-256，不能只凭端口形状相同判断可用。

| 方向 | 名称 | 运行时的大小 |
| --- | --- | --- |
| 输入 | `input_frame` | `[480]` |
| 输入 | `states` | `[45304]` |
| 输入 | `atten_lim_db` | `[]` 或 `[1]` |
| 输出 | `enhanced_audio_frame` | `[480]` |
| 输出 | `new_states` | `[45304]` |
| 输出 | `lsnr` | `[]` 或 `[1]` |

模型可以给一维数组的长度取名，而不写固定数值。加载时会检查维数和已经写明的长度；符号长度不能改变运行时需要的 480 个采样和 45,304 个状态值。对应常量是 `DFN3::SAMPLE_RATE = 48000`、`FRAME_SIZE = 480`、`STATE_SIZE = 45304`。

音频和状态输出会写入事先分配的空间，ONNX Runtime 会检查实际长度。`lsnr` 只检查模型声明的类型和大小，不读取或返回计算结果。如果模型有额外输入输出、数组维数不同，或状态值的含义和顺序不同，就需要修改实现，不能直接使用。

### 选择后端和运行库

只添加 `anr` 依赖不会启用模型处理功能。按需选择 Cargo features：

| 可选功能 | 作用 |
| --- | --- |
| `ort` | 提供 `DFN3`；保留默认 features 时，会在构建中下载并复制 CPU 运行库。 |
| `gl` | 提供 `anr::gl::DFN3`，直接使用原生 OpenGL 4.3 Compute Shader，不依赖 ORT。 |
| `gles` | 提供 `anr::gles::DFN3`，使用真实原生 OpenGL ES 3.1+ Compute 上下文，不依赖 ORT。 |
| `vk` | 提供 `anr::vk::DFN3`，使用原生 Vulkan 1.1 Compute，不依赖 ORT 或 wgpu。 |
| `wgpu` | 提供 `anr::wgpu::DFN3`，使用 wgpu 的硬件 adapter，不依赖 ORT。 |
| `dx` | 提供 `anr::dx::DFN3`，使用原生 Windows Direct3D 12 Compute，不是 DirectML 或 wgpu。 |
| `ort-download-binaries` | 默认下载选项，只在启用 `ort` 后生效，本身不会启用后端。 |
| `ort-load-dynamic` | 加载自备运行库；配合 `default-features = false` 可去掉下载器和 TLS 构建依赖，通过 `ORT_DYLIB_PATH` 指向共享库文件。 |
| `ort-cuda` / `ort-tensorrt` | 编入 CUDA / TensorRT 支持。 |
| `ort-directml` / `ort-coreml` / `ort-xnnpack` | 编入 DirectML / CoreML / XNNPACK 支持。 |

当前固定使用 `ort = 2.0.0-rc.15` 和 API 27，不启用 `ndarray`。默认预编译运行库为 1.31.0，也可使用兼容 API 27 的自备运行库，包括 1.28.0。动态构建需在初始化前设置 `ORT_DYLIB_PATH`，指向库文件而不是目录。

创建环境前先调用 `anr::ort::initialize_runtime()`，让动态库加载错误正常返回；这个函数只准备运行库，不创建或覆盖环境。应用创建一个 `Environment`，供多个降噪实例共享。`DFN3::from_file(&environment, path)` 显式选择内置 CPU provider，以一个线程执行算子并关闭线程池忙等待。GPU features 不会选择 GPU 或安装驱动；需要自定义 provider 时，用 `anr::ort::session::Session::builder(&environment)` 建立 session，再传给 `DFN3::new(session)`。

### 不依赖 ONNX Runtime 的 Compute 后端

CLI 使用 `cargo build -p anr-cli --release --no-default-features --features gl,gles,vk,wgpu,dx` 构建，也可只启用其中部分；运行时明确传 `--backend gl`、`gles`、`vk`、`wgpu` 或 `dx`，不要传 ORT 的 `--provider`。CLI 默认构建、默认后端和应用 Release ZIP 仍使用 ORT。

例如库依赖启用 `default-features = false, features = ["gles"]`，通过 `anr::gles::DFN3::from_file("models/dfn3.onnx")` 创建实例；其余对应 `gl`、`vk`、`wgpu`、`dx` 模块，继续使用 `NoiseReduction`。不需要 ORT 环境、运行库、Python、模型转换或外部 shader 编译器。仅从外置 ONNX 读取参数，并校验 SHA-256 `fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9`，拒绝其他导出版本。

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

`anr-dfn3` 共用固定执行顺序、权重布局、工作线程和 WGSL 算子；Naga 生成桌面 GLSL 4.3、GLSL ES 310、SPIR-V 或 HLSL，wgpu 直接使用 WGSL。GLES 复用 `anr-gl` 的执行器，但必须创建 embedded context，不接受桌面 GL 的 ES 兼容模式；因 GLES 没有 `glGetBufferSubData`，在 GPU fence 完成后只映射当前输出帧。FFT 需要 15,360 字节共享内存，低于 ES 3.1 的 16 KiB 最低要求。

FFT、特征归一化、卷积、分组线性层、五个 GRU、频谱滤波和 overlap-add 全部使用 FP32 Compute Shader，权重和双状态缓冲常驻 GPU。每实例持有工作线程；`run` 等待 GPU 输出，不是非阻塞音频 callback。不返回 `lsnr`，不补偿模型延迟；强度为零保留衰减分支，不表示原音旁路。

| 后端 | 驱动与平台要求 |
| --- | --- |
| `gl` | Windows 桌面 OpenGL 4.3；Linux 硬件 EGL device 需支持桌面 GL 4.3 与 pbuffer，不依赖 X11 `DISPLAY`；不支持 macOS。 |
| `gles` | Windows WGL 驱动需支持原生 ES context；Linux 使用硬件 EGL device/pbuffer，均要求 OpenGL ES 3.1+。本后端未提供 Android、WebGL 或 macOS context。 |
| `vk` | Windows 或 Linux 的硬件 Vulkan 1.1 device，提供 compute queue；不创建窗口或 surface。 |
| `wgpu` | Windows、Linux 或 macOS 上可用的硬件 Vulkan、DX12 或 Metal adapter；不是浏览器后端，不使用软件 fallback。 |
| `dx` | Windows 硬件 D3D12 device 和系统 D3DCompiler 47；不使用 WARP、DirectML，也不要求另装 DXC。 |

初始化失败直接返回错误，不退回 CPU。五种后端均已在原生 Windows / RTX 2080 Ti 上通过连续帧独立 ONNX 数值基准及 raw/WAV、立体声、批处理 CLI 验收；GLES-only 编译、非 GPU 测试与 Clippy 也已在 WSL Ubuntu 24.04 通过。不据此保证跨设备性能或音质；Linux GPU 路径、Android 和 macOS 尚未实跑。

将 `ANR_DFN3_MODEL` 设为模型绝对路径后，运行 `cargo test -p anr-gl -p anr-gles -p anr-vk -p anr-wgpu -p anr-dx -- --include-ignored --nocapture`，按实际硬件选择包。数值回归覆盖强度切换、最后不足一帧、reset、跨线程移交和无效输入不改变状态。`tools/generate_gl_reference.py` 使用开发依赖 `onnx==1.19.1`、`numpy==2.2.6` 一次生成五份基准，不使用 ONNX Runtime。

### 各个包怎么选

Rust 应用通常使用 `anr`；要直接处理文件则安装 `anr-cli`。各后端包将具体实现与公共接口分开。

| 包 | 什么时候用 |
| --- | --- |
| `anr` | 一般使用。提供 `NoiseReduction`、`Error`、`Result`；通过 `ort`、`gl`、`gles`、`vk`、`wgpu` 或 `dx` 启用对应 `DFN3`。 |
| `anr-core` | 只需要 `NoiseReduction`、`Error` 和 `Result`，不希望引入模型运行库。 |
| `anr-ort` | 直接使用 DFN3 的 ONNX Runtime 实现，或使用它提供的同一版本 `ort`。 |
| `anr-gl` | 直接使用固定模型的原生 OpenGL 4.3 DFN3 实现，不依赖 ORT。 |
| `anr-gles` | 原生 OpenGL ES 3.1 DFN3，复用 GL 执行器，不退回桌面 GL 或 CPU。 |
| `anr-vk` | 直接使用原生 Vulkan Compute DFN3，不依赖 ORT 或 wgpu。 |
| `anr-wgpu` | 使用 wgpu Compute DFN3，不依赖 ORT。 |
| `anr-dx` | 使用原生 Windows Direct3D 12 Compute DFN3，不依赖 ORT、DirectML 或 wgpu。 |
| `anr-dfn3` | 固定模型参数、执行顺序、工作线程与 shader 源码；本身不提供设备后端。 |
| `anr-cli` | 安装名为 `anr` 的命令行工具，处理文件、批处理、音频信息查询和 raw PCM 流；默认启用 CPU ORT。 |

`anr-dfn3` 和五个 Compute crate 在源码包中复用本根 README；`anr-core`、`anr-ort`、facade 和 CLI 各有独立 README。Compute 构造器只传模型路径，不接收 ORT 的环境或 `Session`。

`NoiseReduction` 支持 `Box<dyn NoiseReduction>`，降噪器也可以移交给另一个线程（`Send`）。库包不负责音频设备管理、WAV 解码、采样率转换或多声道拆分；CLI 在库外处理文件解码、重采样和多声道。所有包都不自动下载模型。

### 自动发布到 GitHub

[工作流](.github/workflows/ci.yml) 在 Windows x64、Linux x64、macOS arm64/x64 原生 runner 验证。它从固定 Release 下载模型及许可、核对 SHA-256，再解压各平台 ZIP，用控制图和真实 DFN3 实跑；四个平台全部通过后才公开发版。

| 触发事件 | 四个平台通过后的结果 |
| --- | --- |
| push 到 `main` 或现有 `master` | 生成不可变的 `main-<12位SHA>` / `master-<12位SHA>` 预发布，不设为 latest。 |
| push `v<工作区版本>` tag | 发布对应版本；Cargo 预发布后缀仍标为 GitHub prerelease。 |
| pull request 或手动 CI | 只测试和生成 ZIP artifacts，不发布 Release。 |

发布使用 Actions 自动提供的 **`GITHUB_TOKEN`**，映射为 `GH_TOKEN`，仅发布 job 授予 **Contents: write**；无需配置额外仓库 Secret，PR 与构建 job 保持只读。

部分历史提交的 workflow 变更可能触发 **Workflows: write** 授权检查，内置 token 无法获得该权限（[API 说明](https://docs.github.com/en/rest/releases/releases#create-a-release)）。当前策略接受这类发版明确失败，不换用较新的提交，也不移动 tag 或覆盖资产。

先提交并推送工作流改动，再推版本 tag。当前 `Cargo.toml` 对应 `v0.1.1`；tag 与版本不匹配会失败，不生成版本错误的包。该工作流不发布 crates.io，也不会在重跑时覆盖已有公开 Release 或资产。

### 许可

代码采用 **MIT OR Apache-2.0**，使用时任选其一。完整文本见 [LICENSE-MIT](LICENSE-MIT) 和 [LICENSE-APACHE](LICENSE-APACHE)。每个独立发布包附带两份完整文本是本项目的分发选择，不是 Cargo 的强制要求。

本仓库提供的 [模型资产 Release](https://github.com/lona-cn/anr/releases/tag/model-dfn3-fe5eb64fa2e4) 已由维护者授权按 **MIT OR Apache-2.0** 分发，仅限 [清单](.github/models/dfn3.json) 中 SHA-256 对应的文件；ZIP 附带模型许可与说明。其他来源的模型仍需单独确认，不能从 DeepFilterNet 源码的许可推断任意导出文件或权重的权利。

CLI 的 Symphonia 解码器采用 MPL-2.0；rubato、Rust `ort`、ONNX Runtime 及执行 provider 所依赖的库也各有许可要求。源码获取、修改与二进制分发义务见 [第三方说明](crates/anr-cli/THIRD-PARTY-NOTICES.md)，项目双许可不替代这些条款。

版本变更见 [CHANGELOG.md](CHANGELOG.md)，持续维护的发版步骤见 [CI and Release](.github/workflows/ci.yml)。
