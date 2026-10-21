# anr-ort

[English](#english) | [简体中文](#简体中文)

## English

`anr-ort` runs a DeepFilterNet3 (DFN3) model with ONNX Runtime on **48 kHz, mono `f32` samples**. It provides `DFN3` to load the model and process audio in chunks.

It does not read microphones, decode audio files, or resample. Requires the **latest stable Rust toolchain**. Bring your own model: models are not included in the Rust packages or downloaded automatically.

This is specifically the ORT implementation. For OpenGL, GLES, Vulkan, wgpu, or native D3D12 without ONNX Runtime, choose a compute feature of the [`anr` facade](../anr/README.md#compute-backends-without-ort). Those backends use `DFN3::from_file(path)` without an ORT environment and require the fixed model hash; they are not ORT execution providers.

### Denoise your first samples

```toml
[dependencies]
anr-core = "0.1.1"
anr-ort = "0.1.1"
```

The dependency examples use version `0.1.1`. To use a local checkout, replace the version dependencies with `path` dependencies pointing to the crate directories.

```rust,no_run
use anr_core::{NoiseReduction, Result};
use anr_ort::DFN3;

fn main() -> Result<()> {
    anr_ort::initialize_runtime()?;
    let environment = anr_ort::ort::init().build().map_err(anr_core::Error::backend)?;
    let mut denoiser = DFN3::from_file(&environment, "models/dfn3.onnx")?;
    denoiser.set_factor(80.0);

    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    denoiser.run(DFN3::SAMPLE_RATE, &input, &mut output)?;
    denoiser.reset();
    Ok(())
}
```

The example processes silence; replace it with your own samples. The model path is relative to your program's working directory. `run`, `set_factor`, and `reset` come from `anr_core::NoiseReduction`, so import the trait as shown.

### Keep one denoiser per stream

The model uses information from earlier audio, so keep a separate `DFN3` instance for each stream. Call `reset()` before switching recordings or restarting playback; this clears the audio state without changing the strength.

Use chunks that are multiples of 480 samples where possible. The model processes 480 samples, or 10 ms of audio, per frame. A short final frame is padded with zeros, processed as a full frame, and trimmed back to the requested length. The padding still affects later state, so many short calls may produce different results from one call over the whole recording. For live audio, collect samples into multiples of 480 first.

The library does not compensate for model latency and has no end-of-stream flush operation.

### Check the input and output

- Supply at least one input sample.
- Use a sample rate of 48,000 Hz.
- Allocate an output slice with the same length as the input.

If any of these checks fail, the output and processing state stay unchanged. An error during model execution may leave both partly updated; it does not undo work already done. Call `reset()` before reusing the stream after an inference failure; recreate the instance if the underlying runtime or session can no longer run.

### Adjust the strength

`set_factor` accepts `0..=100` and converts it to dB using `round(factor / 100 * 60)`. To set `0..=60` dB directly, use `set_atten_lim_db`; read it with `get_atten_lim_db`.

Out-of-range values are clamped. `NaN` leaves the setting unchanged; positive and negative infinity use the upper and lower bounds respectively.

`set_factor` rounds to whole dB, while `set_atten_lim_db` keeps fractional values. `factor()` converts the actual dB setting back and rounds it, so it may not return exactly what you passed in. These values limit how much the model may attenuate sound; they are not a mix between the original and denoised audio.

`set_factor(0.0)` still executes the ONNX graph. The supplied DFN3 model's zero-attenuation branch is not an audio bypass; skip the reducer in your application if you need unchanged input. The dB setters/getters above are specific to this ORT type, not methods of the shared trait or compute frontends.

### Required model inputs and outputs

`DFN3::from_file(&environment, path)` loads a model using an application-owned `Environment`; `DFN3::new(session)` uses a configured ONNX Runtime session. Both require exactly these three inputs and three outputs, all `f32`. Input order does not matter. `[]` is a scalar; `[1]` is a one-element vector.

| Direction | Name | Runtime shape |
| --- | --- | --- |
| Input | `input_frame` | `[480]` |
| Input | `states` | `[45304]` |
| Input | `atten_lim_db` | `[]` or `[1]` |
| Output | `enhanced_audio_frame` | `[480]` |
| Output | `new_states` | `[45304]` |
| Output | `lsnr` | `[]` or `[1]` |

A one-dimensional tensor may declare a symbolic length. Loading checks the rank and any fixed lengths; symbolic lengths do not change the required 480 audio samples and 45,304 state values at runtime. The constants are `DFN3::SAMPLE_RATE = 48000`, `FRAME_SIZE = 480`, and `STATE_SIZE = 45304`.

Audio and state outputs use preallocated buffers, and ONNX Runtime checks their actual lengths. For `lsnr`, only the declared type and shape are checked; its result is neither read nor returned. Extra inputs or outputs, different ranks, or a different meaning or ordering of state values require changes to the implementation. Passing these checks does not prove the model is correct or that it will sound good on your audio.

Unlike the compute backends, `anr-ort` does not pin the file to one SHA-256 value. It validates the port types/shapes above, but the graph must also use the expected state layout and attenuation semantics. Equal port shapes alone do not establish semantic compatibility.

### Download the CPU runtime or supply your own

`anr_ort::ort` exposes the ONNX Runtime types used by this crate. The project pins `ort = 2.0.0-rc.15` with API 27, without `ndarray`. The default prebuilt runtime is 1.31.0; API-27-compatible self-provided libraries such as 1.28.0 are also supported.

The default `download-binaries` feature downloads and copies the CPU runtime libraries at build time, requiring network access and platform support. To use a runtime you already have, disable default features and enable `load-dynamic`:

```toml
anr-ort = { version = "0.1.1", default-features = false, features = ["load-dynamic"] }
```

Before building an environment, call `anr_ort::initialize_runtime()` to load `ORT_DYLIB_PATH` through the fallible loader. The variable points to an API-27-compatible shared library file, not a directory; unset or empty uses the platform's default name. Failed loads can be retried, and a loaded library is reused. This function does not build an environment or replace the host's settings. Build and keep one `Environment` in your application and share it between models. Disabling defaults removes the downloader and its TLS/OpenSSL build dependencies.

#### Device features do not select a GPU

`cuda`, `tensorrt`, `directml`, `coreml`, and `xnnpack` compile support for the corresponding libraries. They do not select a device or install drivers or other required libraries.

`DFN3::from_file(&environment, path)` explicitly selects the built-in CPU provider, uses one intra-op thread, and disables spinning, environment EP lists, and automatic accelerator selection. It does not require device discovery in minimal builds. For another provider, use `anr_ort::ort::session::Session::builder(&environment)` and pass the session to `DFN3::new(session)`.

In the facade, `ort-directml` compiles support for ORT's DirectML provider; `dx` exposes the separate native D3D12 implementation. Similarly, `gl`, `gles`, `vk`, and `wgpu` are features of the facade, not features of `anr-ort`. Use [`anr-cli`](../anr-cli/README.md) for file decoding, resampling, per-channel processing, and explicit backend/provider selection.

### License

The code is **MIT OR Apache-2.0**; choose either. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).

Check model weights and ONNX files separately for use and redistribution rights. The DeepFilterNet source being dual-licensed does not mean every model from every source has the same license. The Rust `ort` crate, ONNX Runtime, and execution-provider libraries also have their own license and distribution requirements.

## 简体中文

`anr-ort` 用 ONNX Runtime 运行 DeepFilterNet3（DFN3）模型，处理 **48 kHz、单声道的 `f32` 音频采样**。它提供 `DFN3`，供程序加载模型并逐段处理音频。

它不读取麦克风、不解码音频文件，也不转换采样率。需要 **最新 Rust stable 工具链**。模型由你准备，不随 Rust 包提供，也不会自动下载。

本包专门提供 ORT 实现。若要不依赖 ONNX Runtime 的 OpenGL、GLES、Vulkan、wgpu 或原生 D3D12，请选择 [`anr` 的 Compute feature](../anr/README.md#不使用-ort-的-compute-后端)。它们使用不带 ORT 环境的 `DFN3::from_file(path)`，并要求固定模型摘要，不是 ORT execution provider。

### 先完成一次降噪

```toml
[dependencies]
anr-core = "0.1.1"
anr-ort = "0.1.1"
```

依赖示例使用 `0.1.1`。使用本地源码时，可以把版本依赖改成指向对应 crate 目录的 `path` 依赖。

```rust,no_run
use anr_core::{NoiseReduction, Result};
use anr_ort::DFN3;

fn main() -> Result<()> {
    anr_ort::initialize_runtime()?;
    let environment = anr_ort::ort::init().build().map_err(anr_core::Error::backend)?;
    let mut denoiser = DFN3::from_file(&environment, "models/dfn3.onnx")?;
    denoiser.set_factor(80.0);

    let input = [0.0_f32; DFN3::FRAME_SIZE];
    let mut output = [0.0_f32; DFN3::FRAME_SIZE];
    denoiser.run(DFN3::SAMPLE_RATE, &input, &mut output)?;
    denoiser.reset();
    Ok(())
}
```

示例输入是静音，使用时换成自己的音频采样。模型路径相对于程序的工作目录。`run`、`set_factor` 和 `reset` 来自 `anr_core::NoiseReduction`，需要像示例一样先用 `use` 引入。

### 每路连续音频各用一个降噪器

模型处理后面的音频时会用到前面的信息，所以每路音频各用一个 `DFN3` 降噪器。换录音或重新从头播放前调用 `reset()`；它会清除音频状态，保留降噪强度。

每次尽量传入 480 的整数倍。模型一次处理 480 个采样，也就是 10 ms 音频。末尾不足 480 个采样时会补零，按完整的一帧处理，再只返回原有长度的结果。补上的零也会影响后续处理，所以拆成许多短段，结果可能与整段处理不同。实时收到的零散采样，应先积攒到 480 的整数倍。

库不会消除模型带来的时间延迟，也没有在录音结束后额外取出剩余结果的操作。

### 调用前先检查输入和输出

- 输入至少有一个采样。
- 采样率必须是 48,000 Hz。
- 输出数组由你分配，长度必须与输入相同。

这三项检查失败时，输出和处理状态都不会改变。如果运行模型时才出错，输出和状态可能已经部分更新，不会撤销已经完成的处理。推理失败后再次使用前应调用 `reset()`；运行库或 session 已无法执行时，应重新创建实例。

### 调整降噪强度

`set_factor` 接受 `0..=100`，按 `round(factor / 100 * 60)` 换算成分贝（dB）。也可以用 `set_atten_lim_db` 直接设置 `0..=60` dB，用 `get_atten_lim_db` 读取。

超出范围的值会被限制到最近的上限或下限；`NaN` 不改变当前设置，正无穷按上限处理，负无穷按下限处理。

`set_factor` 换算出的分贝值会取整，`set_atten_lim_db` 则保留小数。`factor()` 根据实际分贝设置换算并取整，读回的值可能与传入值略有不同。这些数值限制的是模型压低声音的幅度，不是原音与降噪后音频的混合比例。

`set_factor(0.0)` 仍然运行 ONNX 图，所提供 DFN3 模型的零衰减分支不表示原音旁路。若需要原样返回输入，应由应用跳过降噪器。上述分贝 setter/getter 仅属于本 ORT 类型，不是通用 trait 或 Compute 前端的方法。

### 模型必须有这六个输入和输出

`DFN3::from_file(&environment, path)` 使用应用持有的 `Environment` 加载模型；`DFN3::new(session)` 使用已经建好的 ONNX Runtime session。两种方式都只接受下表的三个输入和三个输出，类型必须为 `f32`，输入排列不限。`[]` 表示单个数，`[1]` 表示只有一个数的一维数组。

| 方向 | 名称 | 运行时的大小 |
| --- | --- | --- |
| 输入 | `input_frame` | `[480]` |
| 输入 | `states` | `[45304]` |
| 输入 | `atten_lim_db` | `[]` 或 `[1]` |
| 输出 | `enhanced_audio_frame` | `[480]` |
| 输出 | `new_states` | `[45304]` |
| 输出 | `lsnr` | `[]` 或 `[1]` |

模型可以给一维数组的长度取名，而不写固定数值。加载时会检查维数和已经写明的长度；符号长度不能改变运行时需要的 480 个采样和 45,304 个状态值。对应常量是 `DFN3::SAMPLE_RATE = 48000`、`FRAME_SIZE = 480`、`STATE_SIZE = 45304`。

音频和状态输出会写入事先分配的空间，ONNX Runtime 会检查实际长度。`lsnr` 只检查模型声明的类型和大小，不读取或返回计算结果。如果模型有额外输入输出、数组维数不同，或状态值的含义和顺序不同，就需要修改实现，不能直接使用。检查通过也不能证明模型本身正确，或降噪效果符合你的需要。

与 Compute 后端不同，`anr-ort` 不将文件限定为一个 SHA-256 摘要；它检查上述端口类型和形状，但模型仍须采用预期的状态布局与衰减语义。端口形状相同不能单独证明语义兼容。

### 默认下载 CPU 运行库，也可以使用自己的运行库

本包通过 `anr_ort::ort` 提供所用版本的运行库类型，固定使用 `ort = 2.0.0-rc.15` 和 API 27，不启用 `ndarray`。默认预编译运行库为 1.31.0，也支持兼容 API 27 的自备库，例如 1.28.0。

默认启用 `download-binaries`，在构建时下载并复制 CPU 使用的 ONNX Runtime，需要网络和目标平台支持。如果已经准备好运行库，可以关闭默认功能，改用 `load-dynamic`：

```toml
anr-ort = { version = "0.1.1", default-features = false, features = ["load-dynamic"] }
```

创建环境前调用 `anr_ort::initialize_runtime()`，通过可返回错误的入口加载 `ORT_DYLIB_PATH`。变量应指向兼容 API 27 的共享库文件，不是目录；未设置或为空时使用平台默认库名。加载失败后可以重试，已加载的库会复用。这个函数不创建环境或替换宿主设置；应用自行创建并持有一个 `Environment`，供模型共享。关闭 defaults 后不再引入下载器及 TLS/OpenSSL 构建依赖。

#### 启用设备支持，不会自动改用 GPU

`cuda`、`tensorrt`、`directml`、`coreml`、`xnnpack` 编入对应库的支持，不会自动选择设备，也不会安装驱动或其他必需的库。

`DFN3::from_file(&environment, path)` 显式选择内置 CPU provider，单线程执行算子，禁用忙等待、环境 EP 列表和自动加速器选择；minimal 构建不需要设备发现 API。若要选其他 provider，用 `anr_ort::ort::session::Session::builder(&environment)` 建立 session，再传给 `DFN3::new(session)`。

facade 的 `ort-directml` 编入 ORT DirectML provider 支持，`dx` 提供另一套原生 D3D12 实现。`gl`、`gles`、`vk`、`wgpu` 同样属于 facade，不是 `anr-ort` 的 features。文件解码、重采样、按声道处理及明确的后端/provider 选择，见 [`anr-cli`](../anr-cli/README.md)。

### 许可

代码采用 **MIT OR Apache-2.0**，使用时任选其一。完整文本见 [LICENSE-MIT](LICENSE-MIT) 和 [LICENSE-APACHE](LICENSE-APACHE)。

模型权重和 ONNX 文件需要单独确认使用与分发权限。DeepFilterNet 源码采用双许可，不代表任何来源的模型都具有相同许可。Rust `ort` 包、ONNX Runtime 及执行 provider 所依赖的库，也各有自己的许可和分发要求。
