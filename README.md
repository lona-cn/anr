# anr

[English](#english) | [简体中文](#简体中文)

## English

`anr` is a Rust audio denoising library and command-line tool. It runs a DeepFilterNet3 (DFN3) model with ONNX Runtime.

The library takes **48 kHz, mono `f32` samples**. For audio files, resampling, and multiple channels, use [`anr-cli`](crates/anr-cli/README.md). The CLI handles files, batches, and raw PCM streams; it does not provide microphone access, virtual audio devices, or drivers.

You'll need **Rust 1.88 or later** and a compatible ONNX model. Models are not included in the Rust packages or downloaded automatically. A model named “DeepFilterNet3” is not necessarily compatible—check its inputs and outputs below.

### Denoise a file

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

The default build may download ONNX Runtime, so it needs network access and a supported platform toolchain. Intel macOS is not covered by the current default ORT prebuilt distribution. CI is configured for native Windows x64, Linux x64, and macOS arm64/x64, including Rust 1.88 checks; the configuration itself is not evidence that those jobs have passed. See the CLI README for recorded local results and platforms not yet tested.

### Use the Rust library

Start with the `anr` crate and enable `ort` to use `DFN3`:

```toml
[dependencies]
anr = { version = "0.1.0", features = ["ort"] }
```

`0.1.0` is the project version, not a claim that it is published on crates.io or that the package name is reserved. To use a local checkout, replace the version dependency with a `path` dependency pointing to the crate directory.

```rust,no_run
use anr::{DFN3, NoiseReduction, Result};

fn main() -> Result<()> {
    let mut denoiser = DFN3::from_file("models/dfn3.onnx")?;
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

`set_factor` accepts `0..=100` and converts it to dB using `round(factor / 100 * 60)`. To set `0..=60` dB directly, use `set_atten_lim_db`; read it with `get_atten_lim_db`.

Out-of-range values are clamped. `NaN` leaves the setting unchanged; positive and negative infinity use the upper and lower bounds respectively.

`set_factor` rounds to whole dB, while `set_atten_lim_db` keeps fractional values. `factor()` converts the actual dB setting back and rounds it, so it may not return exactly what you passed in. These values limit how much the model may attenuate sound; they are not a mix between the original and denoised audio.

#### Try the raw audio example

Run this from the workspace root:

```text
cargo run -p anr --features ort --example denoise -- <model.onnx> <input.f32le> <output.f32le>
```

The input is **48 kHz mono raw audio, not WAV**: four bytes per sample, stored as little-endian IEEE 754 `f32`. The output uses the same format. In this checkout, `<model.onnx>` can be `models/dfn3.onnx`; that file is not included in the Rust packages. Check that the model meets the requirements below and that you have permission to use it.

### Required model inputs and outputs

The model must have exactly these three inputs and three outputs, all `f32`. Names are matched when loading; input order does not matter. Brackets show tensor shapes: `[]` is a scalar, and `[1]` is a one-element vector.

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

### CPU by default; other devices need configuration

Adding `anr` alone does not enable model execution. Choose the Cargo features you need:

| Feature | What it does |
| --- | --- |
| `ort` | Provides `DFN3`. By default, downloads and copies the CPU ONNX Runtime libraries at build time; requires network access and platform support. |
| `ort-load-dynamic` | Loads a runtime you supply. Set `ORT_DYLIB_PATH` to a compatible shared library file. |
| `ort-cuda` / `ort-tensorrt` | Compiles support for CUDA / TensorRT. |
| `ort-directml` / `ort-coreml` / `ort-xnnpack` | Compiles support for DirectML / CoreML / XNNPACK. |

The project pins `ort = 2.0.0-rc.13` (API 27), without `ndarray`. If you supply a runtime, it must support API 27. Set `ORT_DYLIB_PATH` before the process first initializes ONNX Runtime; point it to a library file, not a directory.

`DFN3::from_file` always uses CPU execution with one intra-op thread and disables thread-pool spinning. Enabling a GPU feature does not select the GPU or install CUDA, TensorRT, or drivers. To choose an execution provider, configure one from `anr::ort::ep` with `anr::ort::session::Session::builder()`, then pass the session to `DFN3::new(session)`.

### Which crate should I use?

Use `anr` in most Rust applications, or install `anr-cli` to work with files. The other two crates separate the interface from the ORT implementation.

| Crate | When to use it |
| --- | --- |
| `anr` | The usual starting point. Exports `NoiseReduction`, `Error`, and `Result`; adds `DFN3` with the `ort` feature. |
| `anr-core` | You need only `NoiseReduction`, `Error`, and `Result`, without a model runtime dependency. |
| `anr-ort` | You want the DFN3 ONNX Runtime implementation directly, or access to the same version of `ort`. |
| `anr-cli` | You want the `anr` executable for files, batches, inspection, and raw PCM streams. CPU ORT is enabled by default. |

`NoiseReduction` works as `Box<dyn NoiseReduction>`, and reducers can be moved between threads (`Send`). The three library crates do not manage audio devices, decode WAV, resample, or split multiple channels. The CLI handles file decoding, resampling, and channels outside the library. No package downloads models automatically.

### License

The code is **MIT OR Apache-2.0**; choose either. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE). Including both full license texts in each standalone package is this project's distribution choice, not a Cargo requirement.

Check model weights and ONNX files separately for use and redistribution rights. The DeepFilterNet source being dual-licensed does not mean every model from every source has the same license. The CLI's Symphonia decoder uses MPL-2.0; rubato, the Rust `ort` crate, ONNX Runtime, and execution-provider libraries have their own terms. See the [third-party notices](crates/anr-cli/THIRD-PARTY-NOTICES.md) for source, modification, and binary distribution obligations. The project's dual license does not replace those terms.

See [CHANGELOG.md](CHANGELOG.md) for version changes and [RELEASING.md](RELEASING.md) for maintainer release steps.

## 简体中文

`anr` 提供 Rust 音频降噪库和命令行工具，使用 DeepFilterNet3（DFN3）模型，通过 ONNX Runtime 运行。

库接口处理 **48 kHz、单声道的 `f32` 采样**。音频文件解码、采样率转换和多声道处理可以交给 [`anr-cli`](crates/anr-cli/README.md)。CLI 用于文件、批处理和 raw PCM 流，不提供麦克风访问、虚拟声卡或驱动。

使用前准备好 **Rust 1.88 或更新版本**，以及符合要求的 ONNX 模型。模型不随 Rust 包提供，也不会自动下载；同样叫 DeepFilterNet3 的模型，输入输出也可能不同，请先核对下文要求。

### 直接处理音频文件

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

默认构建可能下载 ONNX Runtime，需要网络和目标平台工具链。当前 ORT 默认预编译分发不包含 Intel macOS。CI 配置了原生 Windows x64、Linux x64、macOS arm64/x64 和 Rust 1.88 检查，但配置文件不代表这些任务已经通过；已有本地结果和未实测平台见 CLI 文档。

### 在 Rust 程序里使用

一般从 `anr` 包开始，启用 `ort` 后就可以使用 `DFN3`：

```toml
[dependencies]
anr = { version = "0.1.0", features = ["ort"] }
```

这里的 `0.1.0` 是项目版本号，不表示它已经在 crates.io 发布，也不表示包名已被保留。使用本地源码时，可以把版本依赖改成指向对应 crate 目录的 `path` 依赖。

```rust,no_run
use anr::{DFN3, NoiseReduction, Result};

fn main() -> Result<()> {
    let mut denoiser = DFN3::from_file("models/dfn3.onnx")?;
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

`set_factor` 接受 `0..=100`，按 `round(factor / 100 * 60)` 换算成分贝（dB）。也可以用 `set_atten_lim_db` 直接设置 `0..=60` dB，用 `get_atten_lim_db` 读取。

超出范围的值会被限制到最近的上限或下限；`NaN` 不改变当前设置，正无穷按上限处理，负无穷按下限处理。

`set_factor` 换算出的分贝值会取整，`set_atten_lim_db` 则保留小数。`factor()` 根据实际分贝设置换算并取整，读回的值可能与传入值略有不同。这些数值限制的是模型压低声音的幅度，不是原音与降噪后音频的混合比例。

#### 用 raw 音频文件试运行

在工作区根目录运行：

```text
cargo run -p anr --features ort --example denoise -- <model.onnx> <input.f32le> <output.f32le>
```

输入文件是 **48 kHz、单声道的原始采样，不是 WAV**：每个采样占 4 字节，以小端序保存为 IEEE 754 `f32`。输出格式相同。在本项目中，`<model.onnx>` 可以写成 `models/dfn3.onnx`；这个模型文件不随 Rust 包提供。使用前请确认模型符合下面的要求，并且你有权使用它。

### 模型必须有这六个输入和输出

模型只能有下表的三个输入和三个输出，类型都必须为 `f32`。加载时按名称匹配，输入的排列顺序不限。方括号表示数组的维度和长度；`[]` 表示单个数，`[1]` 表示只有一个数的一维数组。

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

### 默认用 CPU，其他设备需要自己配置

只添加 `anr` 依赖不会启用模型处理功能。按需选择 Cargo features：

| 可选功能 | 作用 |
| --- | --- |
| `ort` | 提供 `DFN3`。默认在构建时下载并复制 CPU 使用的 ONNX Runtime，需要网络和目标平台支持。 |
| `ort-load-dynamic` | 使用自己准备的 ONNX Runtime；通过 `ORT_DYLIB_PATH` 指向兼容的共享库文件。 |
| `ort-cuda` / `ort-tensorrt` | 编入 CUDA / TensorRT 支持。 |
| `ort-directml` / `ort-coreml` / `ort-xnnpack` | 编入 DirectML / CoreML / XNNPACK 支持。 |

当前固定使用 `ort = 2.0.0-rc.13`（API 27），不启用 `ndarray`。使用自己的运行库时，需要选择兼容 API 27 的版本，并在进程首次初始化 ONNX Runtime 前设置 `ORT_DYLIB_PATH`。这个变量应指向库文件，不是目录。

`DFN3::from_file` 固定使用 CPU，以一个线程执行算子，并关闭线程池的忙等待。启用 GPU 相关功能不会自动改用 GPU，也不会安装 CUDA、TensorRT 或驱动。若要自己选择执行 provider，通过 `anr::ort::session::Session::builder()` 配置 `anr::ort::ep` 中的选项，再把建好的 session 传给 `DFN3::new(session)`。

### 四个包怎么选

Rust 应用通常使用 `anr`；要直接处理文件则安装 `anr-cli`。其余两个包分别提供接口和 ORT 实现。

| 包 | 什么时候用 |
| --- | --- |
| `anr` | 一般使用。提供 `NoiseReduction`、`Error`、`Result`，启用 `ort` 后提供 `DFN3`。 |
| `anr-core` | 只需要 `NoiseReduction`、`Error` 和 `Result`，不希望引入模型运行库。 |
| `anr-ort` | 直接使用 DFN3 的 ONNX Runtime 实现，或使用它提供的同一版本 `ort`。 |
| `anr-cli` | 安装名为 `anr` 的命令行工具，处理文件、批处理、音频信息查询和 raw PCM 流；默认启用 CPU ORT。 |

`NoiseReduction` 支持 `Box<dyn NoiseReduction>`，降噪器也可以移交给另一个线程（`Send`）。三个库包不负责音频设备管理、WAV 解码、采样率转换或多声道拆分；CLI 在库外处理文件解码、重采样和多声道。所有包都不自动下载模型。

### 许可

代码采用 **MIT OR Apache-2.0**，使用时任选其一。完整文本见 [LICENSE-MIT](LICENSE-MIT) 和 [LICENSE-APACHE](LICENSE-APACHE)。每个独立发布包附带两份完整文本是本项目的分发选择，不是 Cargo 的强制要求。

模型权重和 ONNX 文件需要单独确认使用与分发权限。DeepFilterNet 源码采用双许可，不代表任何来源的模型都具有相同许可。CLI 的 Symphonia 解码器采用 MPL-2.0；rubato、Rust `ort` 包、ONNX Runtime 及执行 provider 所依赖的库也各有许可要求。源码获取、修改与二进制分发义务见 [第三方说明](crates/anr-cli/THIRD-PARTY-NOTICES.md)，项目双许可不替代这些条款。

版本变更见 [CHANGELOG.md](CHANGELOG.md)，维护者发布步骤见 [RELEASING.md](RELEASING.md)。
