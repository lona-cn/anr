# anr-core

[English](#english) | [简体中文](#简体中文)

## English

Backend-independent contracts for stateful, single-channel noise reduction. This crate exports `NoiseReduction`, `Error`, and `Result<T, E = Error>`; it contains no denoising implementation, model downloader, audio decoder, resampler, or runtime loader. Requires the latest stable Rust toolchain.

After `0.3.0` is published to crates.io, add:

```toml
[dependencies]
anr-core = "0.3.0"
```

An application chooses an implementation, for example through the [`anr` facade](https://github.com/lona-cn/anr/blob/v0.3.0/crates/anr/README.md). Library code can depend only on the shared interface:

```rust
use anr_core::{NoiseReduction, Result};

pub fn process<R: NoiseReduction + ?Sized>(
    reducer: &mut R,
    sample_rate: u32,
    input: &[f32],
    output: &mut [f32],
) -> Result<()> {
    reducer.run(sample_rate, input, output)
}
```

This accepts concrete implementations and `&mut dyn NoiseReduction`. The trait is object-safe and supports `Box<dyn NoiseReduction>`. Its `Send` bound permits moving an instance between threads; it does not promise `Sync`, concurrent calls, nonblocking execution, or audio-callback safety.

### Trait contract

| Method | Contract |
| --- | --- |
| `run(&mut self, sample_rate: u32, input: &[f32], output: &mut [f32]) -> Result<()>` | Nonempty mono `f32` input; caller-owned output of equal length. Success writes every output sample. |
| `factor(&self) -> f32` | Current strength, in implementation-defined units. |
| `set_factor(&mut self, value: f32)` | Strength range and treatment of non-finite values are implementation-defined. |
| `reset(&mut self)` | Clears accumulated audio state while preserving strength. |

Successive `run` calls belong to one stream. Use a separate reducer per channel or unrelated stream, and reset before starting another recording. Supported sample rates, framing, partial-frame behavior, and latency belong to the implementation. In particular, padding may advance internal state beyond the returned samples.

Input validation errors must leave output and stream state unchanged. Errors after processing starts have implementation-specific guarantees. See the [DFN3 reference](https://github.com/lona-cn/anr/blob/v0.3.0/crates/anr/README.md) for its 48 kHz / 480-sample framing, attenuation limits, and failure handling; those details are not requirements on every implementation of this trait.

### Errors

`Error` implements `Display` and `std::error::Error` and is `#[non_exhaustive]`; matches outside this crate need a wildcard arm.

| Variant | Meaning |
| --- | --- |
| `EmptyInput` | Input contains no samples. |
| `InvalidSampleRate { expected, actual }` | Required and supplied rates, in hertz. |
| `OutputLengthMismatch { input, output }` | Input and output sample counts differ. |
| `InvalidModel(String)` | Model does not meet the implementation's requirements; the string explains why. |
| `Backend(Box<dyn std::error::Error + Send + Sync>)` | Underlying runtime or dependency error. |

`Error::backend(error)` retains the concrete backend error and its chain. Access or downcast it through `std::error::Error::source`.

### License

MIT OR Apache-2.0, at your option: [LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE). Model files, implementations, and runtime dependencies have their own terms.

## 简体中文

`anr-core` 定义有状态、单声道降噪的公共契约，导出 `NoiseReduction`、`Error` 和 `Result<T, E = Error>`。本包不实现降噪，也不下载模型、解码音频、重采样或加载运行库。需要最新 Rust stable 工具链。

`0.3.0` 发布到 crates.io 后，添加以下依赖：

```toml
[dependencies]
anr-core = "0.3.0"
```

应用可通过 [`anr` facade](https://github.com/lona-cn/anr/blob/v0.3.0/crates/anr/README.md) 选择实现；只负责处理音频的库代码可以仅依赖公共接口：

```rust
use anr_core::{NoiseReduction, Result};

pub fn process<R: NoiseReduction + ?Sized>(
    reducer: &mut R,
    sample_rate: u32,
    input: &[f32],
    output: &mut [f32],
) -> Result<()> {
    reducer.run(sample_rate, input, output)
}
```

该函数既接受具体实现，也接受 `&mut dyn NoiseReduction`。trait 支持 `Box<dyn NoiseReduction>`；`Send` 允许在线程之间移交实例，但不保证 `Sync`、并行调用、非阻塞执行或音频回调安全。

### Trait 契约

| 方法 | 契约 |
| --- | --- |
| `run(&mut self, sample_rate: u32, input: &[f32], output: &mut [f32]) -> Result<()>` | 输入为非空的单声道 `f32` 采样；输出由调用方分配且与输入等长。成功时写入全部输出采样。 |
| `factor(&self) -> f32` | 当前强度，单位由实现定义。 |
| `set_factor(&mut self, value: f32)` | 强度范围和非有限值的处理由实现定义。 |
| `reset(&mut self)` | 清除累积音频状态，保留强度。 |

连续的 `run` 调用属于同一路流。不同声道或无关音频应使用独立实例，开始另一段录音前调用 `reset()`。采样率、分帧、尾帧处理和延迟由实现规定；补零可能使内部状态推进超过实际返回的采样数。

输入校验失败必须保持输出和状态不变；开始处理后的错误保证由实现说明。[DFN3 参考](https://github.com/lona-cn/anr/blob/v0.3.0/crates/anr/README.md) 中的 48 kHz / 480 采样分帧、衰减上限和故障处理是具体实现的约定，不是本 trait 对所有算法的要求。

### 错误

`Error` 实现 `Display` 和 `std::error::Error`，标有 `#[non_exhaustive]`，外部代码匹配时须保留通配分支。

| 变体 | 含义 |
| --- | --- |
| `EmptyInput` | 输入没有采样。 |
| `InvalidSampleRate { expected, actual }` | 所需和实际采样率，单位 Hz。 |
| `OutputLengthMismatch { input, output }` | 输入与输出采样数不同。 |
| `InvalidModel(String)` | 模型不符合实现要求，字符串说明原因。 |
| `Backend(Box<dyn std::error::Error + Send + Sync>)` | 底层运行库或依赖错误。 |

`Error::backend(error)` 保留具体错误类型和错误链，可通过 `std::error::Error::source` 获取或向下转型。

### 许可

MIT OR Apache-2.0，任选其一：[LICENSE-MIT](LICENSE-MIT)、[LICENSE-APACHE](LICENSE-APACHE)。模型文件、具体实现及运行库依赖各有自己的许可条件。
