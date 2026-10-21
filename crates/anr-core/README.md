# anr-core

[English](#english) | [简体中文](#简体中文)

## English

`anr-core` defines how to call a noise reducer. It does not implement denoising or load models or ONNX Runtime. Use it when your code needs to work with a denoiser but the application should choose the implementation.

Requires the **latest stable Rust toolchain**. The application can supply ORT, OpenGL, GLES, Vulkan, wgpu, or D3D12 through the `anr` facade; see the [backend overview](../../README.md#compute-backends-without-onnx-runtime). Depending on `anr-core` alone does not enable any implementation or pull in a runtime or GPU driver library.

### Add the dependency

```toml
[dependencies]
anr-core = "0.1.1"
```

The dependency example uses version `0.1.1`. To use a local checkout, replace the version dependency with a `path` dependency pointing to the crate directory.

### Write code that works with any denoiser

```rust
use anr_core::{NoiseReduction, Result};

fn process(
    denoiser: &mut dyn NoiseReduction,
    sample_rate: u32,
    input: &[f32],
    output: &mut [f32],
) -> Result<()> {
    denoiser.run(sample_rate, input, output)
}
```

The application passes its chosen denoiser to `process`; the function does not need to know how the model runs. `NoiseReduction` works as `Box<dyn NoiseReduction>`. `Send` permits moving a reducer to another thread; it does not promise `Sync`, parallel calls on one instance, nonblocking execution, or audio-callback safety.

| Task | Method | Notes |
| --- | --- | --- |
| Process audio | `run(&mut self, sample_rate: u32, input: &[f32], output: &mut [f32]) -> Result<()>` | Supply the sample rate, input samples, and an output slice of the same length. You allocate the output. |
| Read the strength | `factor(&self) -> f32` | The implementation defines what the value means. |
| Set the strength | `set_factor(&mut self, value: f32)` | The implementation defines the range and how out-of-range values are handled. |
| Start another recording | `reset(&mut self)` | Clears the previous audio state but keeps the strength setting. |

#### Nonempty input, equally sized output

Input is a slice of mono `f32` samples. The implementation determines the sample rate and chunk sizes it accepts. Input validation failures must leave the output and processing state unchanged. For errors after processing starts, check the implementation's documentation.

All DFN3 backends in this workspace require 48 kHz mono audio and process 480 samples per frame. A short final frame is padded with zeros, which also advances state. For continuous audio, use multiples of 480 where possible, keep one reducer per channel, and call `reset()` before switching recordings. These are DFN3 implementation choices, not additional requirements on every implementation of this trait.

The shared strength API is `set_factor` / `factor`. DFN3 maps `0..=100` to whole-dB attenuation limits; zero is not a promise to bypass the model. ORT's `set_atten_lim_db` / `get_atten_lim_db` are concrete-type methods, not part of `NoiseReduction` and not exposed by the compute frontends.

This crate does not read audio files, resample, or download models or runtime libraries.

### Handle errors by their cause

`Result<T>` uses this crate's `Error` by default:

| Error | Meaning |
| --- | --- |
| `EmptyInput` | There are no input samples. |
| `InvalidSampleRate { expected, actual }` | The sample rate is wrong. `expected` is the required rate; `actual` is the supplied rate. |
| `OutputLengthMismatch { input, output }` | Input and output lengths differ. The fields contain their respective lengths. |
| `InvalidModel(String)` | The model does not meet the implementation's requirements. The string explains why. |
| `Backend(Box<dyn std::error::Error + Send + Sync>)` | The implementation or one of its dependencies returned an error. |

`Error` is `#[non_exhaustive]`: new variants may be added. Include a wildcard `_` arm when matching it.

When writing a denoiser, use `Error::backend(error)` to preserve the original error. Applications can inspect it through `std::error::Error::source`.

### License

The code is **MIT OR Apache-2.0**; choose either. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE). Check licenses for model files, denoising implementations, and their dependencies separately.

## 简体中文

`anr-core` 只规定怎样调用降噪器，不提供降噪实现，也不加载模型或 ONNX Runtime。如果你的代码只需要调用降噪器，而由应用决定使用哪种实现，就选这个包。

需要 **最新 Rust stable 工具链**。应用可通过 `anr` 选择 ORT、OpenGL、GLES、Vulkan、wgpu 或 D3D12，参见[后端说明](../../README.md#简体中文)。只依赖 `anr-core` 不会启用任何实现，也不会引入推理运行库或 GPU 驱动库。

### 添加依赖

```toml
[dependencies]
anr-core = "0.1.1"
```

依赖示例使用 `0.1.1`。使用本地源码时，可以把版本依赖改成指向对应 crate 目录的 `path` 依赖。

### 用同一种方式调用降噪器

```rust
use anr_core::{NoiseReduction, Result};

fn process(
    denoiser: &mut dyn NoiseReduction,
    sample_rate: u32,
    input: &[f32],
    output: &mut [f32],
) -> Result<()> {
    denoiser.run(sample_rate, input, output)
}
```

应用把选好的降噪器传给 `process`，函数不需要知道模型怎样运行。`NoiseReduction` 支持 `Box<dyn NoiseReduction>`。`Send` 允许将实例移交给另一个线程，但不保证 `Sync`、同一实例的并行调用、非阻塞执行或音频 callback 安全。

| 要做什么 | 方法 | 说明 |
| --- | --- | --- |
| 处理音频 | `run(&mut self, sample_rate: u32, input: &[f32], output: &mut [f32]) -> Result<()>` | 传入采样率、输入采样和同样长度的输出数组；输出数组由你分配。 |
| 查看强度 | `factor(&self) -> f32` | 数值的含义由具体实现说明。 |
| 调整强度 | `set_factor(&mut self, value: f32)` | 可用范围，以及超出范围时怎样处理，由具体实现说明。 |
| 开始另一段录音 | `reset(&mut self)` | 清除上一段音频的处理状态，保留降噪强度。 |

#### 输入必须非空，输出必须与输入等长

输入使用单声道 `f32` 采样。采样率要求，以及一次应传入多少采样，要看所用实现。输入检查失败时，输出和处理状态都不能改变；处理开始后才发生的错误，则按具体实现的说明处理。

工作区内所有 DFN3 后端都要求 48 kHz 单声道，每帧处理 480 个采样；尾帧补零也会推进状态。连续音频应尽量按 480 的整数倍调用，每声道使用独立实例，换录音前调用 `reset()`。这些是 DFN3 的实现约定，不是本 trait 对所有降噪算法额外规定的条件。

通用强度接口是 `set_factor` / `factor`。DFN3 将 `0..=100` 映射为取整的分贝衰减上限，零不保证旁路。ORT 的 `set_atten_lim_db` / `get_atten_lim_db` 属于具体类型，不在 `NoiseReduction` 中，Compute 前端也不提供这两个方法。

本包不读取音频文件、不转换采样率，也不下载模型或运行库。

### 根据错误原因决定怎样处理

`Result<T>` 默认使用本包的 `Error`：

| 错误 | 含义 |
| --- | --- |
| `EmptyInput` | 输入没有任何采样。 |
| `InvalidSampleRate { expected, actual }` | 采样率不符；`expected` 是所需值，`actual` 是传入值。 |
| `OutputLengthMismatch { input, output }` | 输入输出长度不同；两个字段分别记录它们的长度。 |
| `InvalidModel(String)` | 模型不符合所用实现的要求，字符串说明原因。 |
| `Backend(Box<dyn std::error::Error + Send + Sync>)` | 具体实现或它依赖的库返回了错误。 |

`Error` 标有 `#[non_exhaustive]`，以后可能新增错误类型。使用 `match` 时需要保留 `_` 分支，处理尚未列出的情况。

编写降噪实现时，可以用 `Error::backend(error)` 保留原始错误。应用可以通过 `std::error::Error::source` 查看它的来源。

### 许可

代码采用 **MIT OR Apache-2.0**，使用时任选其一。完整文本见 [LICENSE-MIT](LICENSE-MIT) 和 [LICENSE-APACHE](LICENSE-APACHE)。模型文件、降噪实现及其依赖的库，需要分别确认许可。
