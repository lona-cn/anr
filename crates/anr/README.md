# anr

[English](#english) | [简体中文](#简体中文)

## English

`anr` provides a Rust interface to six DFN3 backends: ORT, OpenGL, GLES, Vulkan, wgpu, and D3D12. It processes **48 kHz, mono `f32` samples** with equally sized input and output. No backend is enabled by default. With `ort`, use `anr::DFN3`; with another backend, use its module, such as `anr::wgpu::DFN3`.

The library does not read microphones, decode WAV, or resample audio. You'll need the **latest stable Rust toolchain** and a model file of your own.

### Choose a backend

| Cargo feature | Backend | Type |
| --- | --- | --- |
| `ort` | ONNX Runtime | `anr::DFN3` |
| `gl` | OpenGL | `anr::gl::DFN3` |
| `gles` | OpenGL ES | `anr::gles::DFN3` |
| `vk` | Vulkan | `anr::vk::DFN3` |
| `wgpu` | wgpu | `anr::wgpu::DFN3` |
| `dx` | Direct3D 12 | `anr::dx::DFN3` |

To build without ORT, set `default-features = false` and list the backend you need under `features`. For example:

```toml
[dependencies]
anr = { version = "0.1.1", default-features = false, features = ["wgpu"] }
```

List the backends your program needs under `features`. Cargo includes all of them; your program must choose which one to use. The five non-ORT backends support only the project's specified DFN3 model, not arbitrary ONNX files. Hardware and system requirements are below.

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
    let mut denoiser = DFN3::from_file(&environment, "models/dfn3.onnx")?;
    denoiser.set_factor(80.0);

    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    denoiser.run(DFN3::SAMPLE_RATE, &input, &mut output)?;
    denoiser.reset();
    Ok(())
}
```

The example processes silence; replace it with your own samples. The model path is relative to your program's working directory. Models are not included in the Rust packages or downloaded automatically.

### Keep one denoiser per stream

Reuse the same `DFN3` for continuous audio, and don't share it between unrelated streams. The model uses information from earlier audio. Call `reset()` before switching recordings or restarting playback; this clears the audio state without changing the strength.

Use chunks that are multiples of 480 samples where possible. `DFN3::FRAME_SIZE = 480`, or 10 ms of audio. `run` accepts any nonzero length, but pads a short final frame with zeros, processes the full frame, and returns only the requested samples. The zeros still affect later state, so many short calls may produce different results from one call over the whole recording.

The library does not compensate for model latency and has no end-of-stream flush operation. Your application must split multichannel audio before processing it.

### Check the input and output

- Supply at least one input sample.
- Use a sample rate of 48,000 Hz.
- Allocate an output slice with the same length as the input.

If any of these checks fail, the output and processing state stay unchanged. An error during model execution may leave both partly updated; it does not undo work already done.

Without a backend feature, `anr` exports only `NoiseReduction`, `Error`, and `Result`. The default `ort-download-binaries` option does not enable ORT by itself. `NoiseReduction` works as `Box<dyn NoiseReduction>`; `Send` allows moving a reducer to another thread but does not guarantee concurrent calls or nonblocking execution.

### Adjust the strength

All DFN3 implementations expose `set_factor(0..=100)` through `NoiseReduction`, mapped to dB as `round(factor / 100 * 60)`. Only `anr::ort::DFN3` additionally provides `set_atten_lim_db` / `get_atten_lim_db` for direct `0..=60` dB access. Those methods are not part of the trait or compute API.

Out-of-range values are clamped. `NaN` leaves the setting unchanged; positive and negative infinity use the upper and lower bounds respectively.

`set_factor` rounds to whole dB; the ORT-only dB setter preserves fractional values. `factor()` rounds the actual setting back to `0..=100`, so it may not return exactly what you passed in. Strength is an attenuation limit, not a dry/wet mix; zero still runs the supplied model and is not an audio bypass.

### Try the ORT raw audio example

Run this from the project root:

```text
cargo run -p anr --features ort --example denoise -- <model.onnx> <input.f32le> <output.f32le>
```

The input is **48 kHz mono raw audio, not WAV**: four bytes per sample, stored as little-endian IEEE 754 `f32`. The output uses the same format. This command requires the source checkout; installing the crate alone does not give you the whole project.

### ORT model requirements

For ORT, `DFN3::from_file(&environment, path)` loads a model file using an application-owned `Environment`; `DFN3::new(session)` uses a configured session. Both require exactly these three inputs and three outputs, all `f32`. Input order does not matter. `[]` is a scalar; `[1]` is a one-element vector. Compute constructors take only `path` and require the fixed model hash below.

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
- `ort-download-binaries`: the default, weak download option. It does not activate `anr-ort` by itself, so the default crate still depends only on `anr-core`.
- `ort-load-dynamic`: loads your runtime. Disable defaults to omit the downloader and TLS dependencies. Set `ORT_DYLIB_PATH` to a compatible shared library file before initialization.
- `ort-cuda`, `ort-tensorrt`, `ort-directml`, `ort-coreml`, `ort-xnnpack`: compile support for the corresponding libraries. They do not select a GPU or install drivers or other required libraries.

The project pins `ort = 2.0.0-rc.15` with API 27, without `ndarray`. The default prebuilt runtime is 1.31.0; compatible API-27 libraries such as 1.28.0 also work. `anr::ort` exposes the same ONNX Runtime types.

```toml
anr = { version = "0.1.1", default-features = false, features = ["ort-load-dynamic"] }
```

Call `anr::ort::initialize_runtime()` before `anr::ort::init().build()` to return dynamic-loading failures instead of entering the lazy loader. The helper does not build an environment; your application owns and configures it. Share that `Environment` across denoisers. `DFN3::from_file(&environment, path)` explicitly selects the built-in CPU provider, one intra-op thread, and no spinning. To choose another provider, use `anr::ort::session::Session::builder(&environment)` and pass the session to `DFN3::new(session)`.

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

The original external ONNX file supplies parameters only and must have SHA-256 `fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9`. No model conversion, Python, or external shader compiler is required. Other exports and unsupported drivers fail explicitly. Strength zero preserves the model's attenuation branch; it is not an audio bypass.

### License

The code is **MIT OR Apache-2.0**; choose either. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).

Check model weights and ONNX files separately for use and redistribution rights. The DeepFilterNet source being dual-licensed does not mean every model from every source has the same license. The Rust `ort` crate, ONNX Runtime, and execution-provider libraries also have their own license and distribution requirements.

## 简体中文

`anr` 提供六种 DFN3 后端：ORT、OpenGL、GLES、Vulkan、wgpu 和 D3D12。它处理 **48 kHz、单声道 `f32` 采样**，输入输出等长。默认不启用任何后端。启用 `ort` 后可使用 `anr::DFN3`；启用其他后端后，从对应模块使用，例如 `anr::wgpu::DFN3`。

它不读取麦克风、不解码 WAV，也不转换采样率。需要 **最新 Rust stable 工具链**，以及自己准备的模型文件。

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

在 `features` 中填写要用的后端。写多个就会同时编入多个，程序需要自行决定使用哪一个；它不会自动切换。五种非 ORT 后端只支持本项目指定的 DFN3 模型，不能拿来运行任意 ONNX 文件。硬件和系统要求见下文。

### ORT 入门

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

    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    denoiser.run(DFN3::SAMPLE_RATE, &input, &mut output)?;
    denoiser.reset();
    Ok(())
}
```

示例输入是静音，使用时换成自己的音频采样。模型路径相对于程序的工作目录；模型不随 Rust 包提供，也不会自动下载。

### 每路连续音频各用一个降噪器

连续录音要一直使用同一个 `DFN3` 降噪器，不要让几路无关音频轮流共用。模型处理后面的音频时会用到前面的信息。换录音或重新从头播放前调用 `reset()`；它会清除音频状态，保留降噪强度。

每次尽量传入 480 的整数倍。`DFN3::FRAME_SIZE = 480`，也就是 10 ms 音频。`run` 接受任意非零长度，但末尾不足 480 个采样时会补零，按完整的一帧处理，再只返回原有长度的结果。补上的零也会影响后续处理，所以拆成许多短段，结果可能与整段处理不同。

库不会消除模型带来的时间延迟，也没有在录音结束后额外取出剩余结果的操作。多声道音频需要由应用先分开处理。

### 调用前先检查输入和输出

- 输入至少有一个采样。
- 采样率必须是 48,000 Hz。
- 输出数组由你分配，长度必须与输入相同。

这三项检查失败时，输出和处理状态都不会改变。如果运行模型时才出错，输出和状态可能已经部分更新，不会撤销已经完成的处理。

未启用后端 feature 时，`anr` 只导出 `NoiseReduction`、`Error`、`Result`；默认的 `ort-download-binaries` 不会单独启用 ORT。`NoiseReduction` 支持 `Box<dyn NoiseReduction>`；`Send` 允许跨线程移交实例，不保证并行调用或非阻塞执行。

### 调整降噪强度

所有 DFN3 实现都通过 `NoiseReduction` 提供 `set_factor(0..=100)`，按 `round(factor / 100 * 60)` 换算成分贝。只有 `anr::ort::DFN3` 另外提供 `set_atten_lim_db` / `get_atten_lim_db`，用于直接访问 `0..=60` dB；它们不属于通用 trait 或 Compute API。

超出范围的值会被限制到最近的上限或下限；`NaN` 不改变当前设置，正无穷按上限处理，负无穷按下限处理。

`set_factor` 会将分贝值取整，ORT 专属的分贝 setter 保留小数。`factor()` 根据实际设置换算并取整，可能与传入值略有不同。强度是衰减上限，不是原音与处理结果的混合比例；零仍运行所提供的模型，不表示原音旁路。

### 用 ORT raw 音频示例试运行

在项目根目录执行：

```text
cargo run -p anr --features ort --example denoise -- <model.onnx> <input.f32le> <output.f32le>
```

输入文件是 **48 kHz、单声道的原始采样，不是 WAV**：每个采样占 4 字节，以小端序保存为 IEEE 754 `f32`。输出格式相同。这个命令需要本项目源码，只安装 Rust 包不会得到整个项目。

### ORT 模型要求

ORT 的 `DFN3::from_file(&environment, path)` 使用应用持有的 `Environment` 加载模型，`DFN3::new(session)` 使用已配置的 session。两种方式都只接受下表的三个输入和三个输出，类型为 `f32`，输入排列不限。`[]` 是单个数，`[1]` 是单元素的一维数组。Compute 构造器只传 `path`，并要求后文指定的固定模型摘要。

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
- `ort-download-binaries`：默认启用的弱下载选项，不会单独激活 `anr-ort`，所以默认 crate 仍只依赖 `anr-core`。
- `ort-load-dynamic`：加载自备运行库。关闭 defaults 可去掉下载器和 TLS 依赖；初始化前将 `ORT_DYLIB_PATH` 设为兼容的共享库文件。
- `ort-cuda`、`ort-tensorrt`、`ort-directml`、`ort-coreml`、`ort-xnnpack`：编入对应库的支持，不会自动选择 GPU，也不会安装驱动或其他必需的库。

当前固定使用 `ort = 2.0.0-rc.15` 和 API 27，不启用 `ndarray`。默认预编译运行库为 1.31.0，也可用兼容 API 27 的库，例如 1.28.0。`anr::ort` 提供同一版本的运行库类型。

```toml
anr = { version = "0.1.1", default-features = false, features = ["ort-load-dynamic"] }
```

在 `anr::ort::init().build()` 前调用 `anr::ort::initialize_runtime()`，让动态加载失败正常返回，避免进入上游的隐式加载路径。这个函数不创建环境，环境由应用持有和配置，多个降噪实例共享同一个 `Environment`。`DFN3::from_file(&environment, path)` 显式选择内置 CPU provider，以单线程执行算子并禁用忙等待。若要选其他 provider，用 `anr::ort::session::Session::builder(&environment)` 建立 session，再传给 `DFN3::new(session)`。

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

原始外置 ONNX 只提供参数，要求 SHA-256 为 `fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9`。无需模型转换、Python 或外部 shader 编译器。其他模型或不支持的驱动明确报错。强度为零保留原模型的衰减分支，不表示原音旁路。

### 许可

代码采用 **MIT OR Apache-2.0**，使用时任选其一。完整文本见 [LICENSE-MIT](LICENSE-MIT) 和 [LICENSE-APACHE](LICENSE-APACHE)。

模型权重和 ONNX 文件需要单独确认使用与分发权限。DeepFilterNet 源码采用双许可，不代表任何来源的模型都具有相同许可。Rust `ort` 包、ONNX Runtime 及执行 provider 所依赖的库，也各有自己的许可和分发要求。
