# 第三方组件与分发说明

`anr-cli` 自有代码采用 MIT OR Apache-2.0；这不改变依赖、运行库或模型的许可。本文件说明主要边界和源码获取方法，不代替各组件的完整许可，也不是任意 feature 组合的完整二进制许可清单。发布者应依据实际 Cargo.lock、启用 features、链接的 ONNX Runtime 构建和所附动态库，保留对应版权、许可及 NOTICE/ThirdPartyNotices 文件。

## 音频解码：Symphonia 0.5.5（MPL-2.0）

Symphonia 是 Philip Deljanov 及其贡献者的纯 Rust 容器/codec 实现。当前 CLI 启用 `all`，不通过系统 FFmpeg 解码。

- 上游源码与原始版权通知：[pdeljanov/Symphonia](https://github.com/pdeljanov/Symphonia)。
- 完整许可：[Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/)。
- 精确版本源码入口：[symphonia 0.5.5 源码](https://docs.rs/crate/symphonia/0.5.5/source/)，或下载 [crates.io 源码包](https://crates.io/api/v1/crates/symphonia/0.5.5/download)。

同版本组件的源码分别位于以下入口；不是只提供顶层 façade 就足以覆盖静态链接进去的 codec：

- [symphonia-core](https://docs.rs/crate/symphonia-core/0.5.5/source/)、[symphonia-metadata](https://docs.rs/crate/symphonia-metadata/0.5.5/source/)、[symphonia-utils-xiph](https://docs.rs/crate/symphonia-utils-xiph/0.5.5/source/)。
- [symphonia-bundle-flac](https://docs.rs/crate/symphonia-bundle-flac/0.5.5/source/)、[symphonia-bundle-mp3](https://docs.rs/crate/symphonia-bundle-mp3/0.5.5/source/)。
- [symphonia-codec-aac](https://docs.rs/crate/symphonia-codec-aac/0.5.5/source/)、[symphonia-codec-adpcm](https://docs.rs/crate/symphonia-codec-adpcm/0.5.5/source/)、[symphonia-codec-alac](https://docs.rs/crate/symphonia-codec-alac/0.5.5/source/)、[symphonia-codec-pcm](https://docs.rs/crate/symphonia-codec-pcm/0.5.5/source/)、[symphonia-codec-vorbis](https://docs.rs/crate/symphonia-codec-vorbis/0.5.5/source/)。
- [symphonia-format-caf](https://docs.rs/crate/symphonia-format-caf/0.5.5/source/)、[symphonia-format-isomp4](https://docs.rs/crate/symphonia-format-isomp4/0.5.5/source/)、[symphonia-format-mkv](https://docs.rs/crate/symphonia-format-mkv/0.5.5/source/)、[symphonia-format-ogg](https://docs.rs/crate/symphonia-format-ogg/0.5.5/source/)、[symphonia-format-riff](https://docs.rs/crate/symphonia-format-riff/0.5.5/source/)。

MPL-2.0 是文件级许可。其第 3.1–3.4 条要求，分发被覆盖源码（包括对这些文件的修改）时仍以 MPL-2.0 提供并保留实质性许可通知；分发包含这些组件的 executable 时，也必须及时以合理方式提供对应的 Source Code Form，并告诉接收者如何取得，收费不超过分发成本。它不要求把仅与其组合、位于独立文件中的全部 CLI 代码改成 MPL，但不能以项目双许可限制接收者对 MPL 源码的权利。

未修改的 registry 组件可以从上述精确版本入口取得。若构建者使用修改版或 patch，**必须另行提供实际使用的修改后源码**，只有上游链接不够；应随分发提供源码归档或有效下载位置，并保留适用通知。需要离线归档时，可从工作区用 `cargo vendor --locked <源码归档目录>` 获取锁定的依赖源码，再核对实际构建中的 patch 和版权/许可文件。二进制包应携带本说明、MPL 完整文本及对应源码获取说明；不能仅附项目的 MIT/Apache 文本。

音频格式实现的软件许可也不是针对所有地区的 codec 专利或其他权利保证；具体使用和分发场景仍由分发者核实。

## 重采样、WAV 和其他 Rust 依赖

| 组件 | 许可 | 上游 |
| --- | --- | --- |
| rubato 0.16.2 | MIT | [HEnquist/rubato](https://github.com/HEnquist/rubato)，[精确版本源码](https://docs.rs/crate/rubato/0.16.2/source/) |
| hound 3.5.1 | Apache-2.0 | [ruuda/hound](https://github.com/ruuda/hound)，[精确版本源码](https://docs.rs/crate/hound/3.5.1/source/) |
| RustFFT（rubato 的 FFT 依赖链） | MIT OR Apache-2.0 | [ejmahler/RustFFT](https://github.com/ejmahler/RustFFT) |
| clap、anyhow、ctrlc、tempfile、serde、serde_json | 各自声明 MIT OR Apache-2.0 | [clap](https://github.com/clap-rs/clap)、[anyhow](https://github.com/dtolnay/anyhow)、[ctrlc](https://github.com/Detegr/rust-ctrlc)、[tempfile](https://github.com/Stebalien/tempfile)、[serde](https://github.com/serde-rs/serde)、[serde_json](https://github.com/serde-rs/json) |

以上不省略传递依赖义务；以实际解析版本的 package 源码及许可为准。尤其 MIT 的版权和许可文本、Apache-2.0 的许可及适用 NOTICE 不可用项目自身的版权通知替换。

## Rust ORT 绑定与 ONNX Runtime 是两个组件

- Rust `ort` / `ort-sys` 固定为 **2.0.0-rc.13**，声明 **MIT OR Apache-2.0**，来自 [pykeio/ort](https://github.com/pykeio/ort)。其 [ort 源码](https://docs.rs/crate/ort/2.0.0-rc.13/source/) 与 [ort-sys 源码](https://docs.rs/crate/ort-sys/2.0.0-rc.13/source/) 含独立版权和许可。
- Microsoft **ONNX Runtime** 是实际执行推理的原生运行库，不是 Rust 绑定的另一名称。其 [上游源码](https://github.com/microsoft/onnxruntime)、[v1.28.0 MIT 许可](https://github.com/microsoft/onnxruntime/blob/v1.28.0/LICENSE) 和 [ThirdPartyNotices](https://github.com/microsoft/onnxruntime/blob/v1.28.0/ThirdPartyNotices.txt) 需独立处理。默认下载的打包构建、自建库和用户通过 `ORT_DYLIB_PATH` 选择的库可能具有不同组件集合，必须核对实际构建附带文件。
- CUDA、TensorRT、DirectML、CoreML、XNNPACK 等 provider 及其驱动、SDK、系统框架或随附库各有适用条款。选择 Cargo feature 不代表取得其再分发许可，也不表示这些库已被安装。分发者只应携带有权分发的运行库及其通知，不把 Microsoft、NVIDIA、Apple 或其他供应商组件列为 anr contributors 自有代码。

## 模型权重与测试控制图

DFN3 模型权重和导出 ONNX 文件不随 CLI crate 提供，也不自动下载。权重、训练数据、导出过程及使用限制需按实际来源确认；[DeepFilterNet 源码仓库](https://github.com/Rikorose/DeepFilterNet) 的双许可不能推出任意下载模型的使用或再分发权限。

`tests/fixtures` 中本项目自制的微型 ONNX 控制图用于验证输入输出、状态和错误处理，属于项目测试材料，不含用户 DFN3 权重，不宣称具有真实降噪能力。CI 使用这些控制图，不重新分发工作区 `models/` 中的用户模型。
