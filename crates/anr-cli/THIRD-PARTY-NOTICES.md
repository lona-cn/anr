# 第三方组件与分发说明

`anr-cli` 自有代码采用 MIT OR Apache-2.0；这不改变依赖、运行库或模型的许可。本文件说明主要边界和源码获取方法，不代替各组件的完整许可，也不是任意 feature 组合的完整二进制许可清单。发布者应依据实际 Cargo.lock、启用 features、链接或动态加载的运行库和所附动态库，保留对应版权、许可及 NOTICE/ThirdPartyNotices 文件。

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

## ONNX 消息解析：onnx-protobuf 0.2.3（MPL-2.0）

启用 ORT 时，加载前的内存图转换使用未修改的 `onnx-protobuf` 0.2.3；它提供完整 ONNX 消息，并保留未知 protobuf 字段。

- 精确版本源码：[docs.rs 源码](https://docs.rs/crate/onnx-protobuf/0.2.3/source/)，或 [crates.io 源码包](https://crates.io/api/v1/crates/onnx-protobuf/0.2.3/download)。
- 上游：[oovm/Airs 的 onnx-protobuf](https://github.com/oovm/Airs/tree/dev/projects/onnx-protobuf)。
- 完整许可：[Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/)。与上述 Symphonia 相同的文件级源码获取、修改和二进制分发义务也适用于此组件，不能只提供项目的双许可。
- 生成的消息要求精确的 `protobuf` 3.4.0 运行库；该组件采用 MIT，源码和许可见 [protobuf 3.4.0](https://docs.rs/crate/protobuf/3.4.0/source/)。


## 重采样、WAV 和其他 Rust 依赖

| 组件 | 许可 | 上游 |
| --- | --- | --- |
| rubato 0.16.2 | MIT | [HEnquist/rubato](https://github.com/HEnquist/rubato)，[精确版本源码](https://docs.rs/crate/rubato/0.16.2/source/) |
| hound 3.5.1 | Apache-2.0 | [ruuda/hound](https://github.com/ruuda/hound)，[精确版本源码](https://docs.rs/crate/hound/3.5.1/source/) |
| RustFFT（rubato 的 FFT 依赖链） | MIT OR Apache-2.0 | [ejmahler/RustFFT](https://github.com/ejmahler/RustFFT) |
| file-id 0.2.3 | MIT OR Apache-2.0 | [notify-rs/notify](https://github.com/notify-rs/notify)，[精确版本源码](https://docs.rs/crate/file-id/0.2.3/source/) |
| clap、anyhow、ctrlc、tempfile、serde、serde_json | 各自声明 MIT OR Apache-2.0 | [clap](https://github.com/clap-rs/clap)、[anyhow](https://github.com/dtolnay/anyhow)、[ctrlc](https://github.com/Detegr/rust-ctrlc)、[tempfile](https://github.com/Stebalien/tempfile)、[serde](https://github.com/serde-rs/serde)、[serde_json](https://github.com/serde-rs/json) |

以上不省略传递依赖义务；以实际解析版本的 package 源码及许可为准。尤其 MIT 的版权和许可文本、Apache-2.0 的许可及适用 NOTICE 不可用项目自身的版权通知替换。

## 可选 Compute 后端

只启用 `gl`、`gles`、`vk`、`wgpu`、`dx` 的构建不链接 Rust ORT 或 ONNX Runtime。`gl` 和 `gles` 共用 [glow](https://github.com/grovesNL/glow)、[glutin](https://github.com/rust-windowing/glutin) 和 raw-window-handle 调用系统驱动、管理原生上下文；GLES 生成 GLSL ES 310 并要求真实 ES context。`vk` 使用 [ash](https://github.com/ash-rs/ash)；`wgpu` 使用 [gfx-rs/wgpu](https://github.com/gfx-rs/wgpu) 的原生硬件后端；`dx` 使用 [windows-rs](https://github.com/microsoft/windows-rs) 调用系统 D3D12、DXGI 和 D3DCompiler 47，不是 DirectML。驱动与系统库不随 crate 安装或打包，分发者应另行核对运行环境要求。

共用 WGSL 算子通过 [Naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga) 翻译为 GLSL、GLSL ES、SPIR-V 或 HLSL。参数解码使用 [prost](https://github.com/tokio-rs/prost)，摘要使用 [sha2](https://github.com/RustCrypto/hashes)，字节视图使用 [bytemuck](https://github.com/Lokathor/bytemuck)。各包及传递依赖保留自己的许可和版权通知；无 ORT 分发仍应按实际 Cargo.lock 收集对应文本，不能用项目许可代替。模型权重的授权与代码依赖分别处理。

## 原生 CUDA、TensorRT、OpenVINO 与 OpenCL

`cuda`、`tensorrt`、`openvino`、`opencl` 是独立于 ORT 的可选后端；它们不启用 Rust `ort` 或 ONNX Runtime。`ort-cuda`、`ort-tensorrt` 则仍属于 ORT provider 链。以下边界适用于实际启用和分发的组件，不意味着所有库都在默认 CPU ORT ZIP 中：

| 组件 | 许可边界与一手来源 |
| --- | --- |
| `cudarc` 0.19.10 | 原生 CUDA 使用的 Rust Driver/NVRTC 动态绑定，**MIT OR Apache-2.0**；[上游](https://github.com/chelsea0x3b/cudarc)、[MIT](https://github.com/chelsea0x3b/cudarc/blob/main/LICENSE-MIT)、[Apache-2.0](https://github.com/chelsea0x3b/cudarc/blob/main/LICENSE-APACHE)、[精确版本源码](https://docs.rs/crate/cudarc/0.19.10/source/)。绑定许可不覆盖 NVIDIA 库。 |
| NVIDIA CUDA Driver、NVRTC / NVRTC builtins、CUDA Runtime | 本项目的 CUDA kernel 源码不改变 NVIDIA 原生组件的许可。CUDA 后端运行时加载 Driver 和 NVRTC；TensorRT shim 还链接 CUDA Runtime 等库。按实际 SDK/运行库版本核对 [CUDA Toolkit EULA、Supplement 与 Attachment A](https://docs.nvidia.com/cuda/eula/index.html)，只分发被明确允许且满足分发条件的文件，保留适用许可和通知。驱动应通过厂商支持的安装渠道提供；不能由 Rust 绑定双许可推断驱动、整个 Toolkit 或任意 DLL 可自由再分发。 |
| NVIDIA TensorRT | `anr-tensorrt` 编译本项目私有 C ABI shim，神经网络使用 TensorRT standard layers，DSP 使用 CUDA；不使用 ORT、ONNX parser 或自定义推理 plugin，但实际 `nvinfer` 等 vendor runtime 仍受 [TensorRT SDK License Agreement 及 Supplement](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html) 和随 SDK 的通知约束。[TensorRT 开源仓库的 Apache-2.0 许可](https://github.com/NVIDIA/TensorRT/blob/main/LICENSE) 不可泛化为所有二进制 SDK 的许可。 |
| OpenVINO Runtime、ONNX frontend、设备 plugins | `anr-openvino` 的私有 shim 调用原生 C++ Runtime。OpenVINO 项目采用 **Apache-2.0**，见 [上游 LICENSE](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE)；分发时保留实际所用版本的许可、版权、适用 NOTICE 和 [third-party-programs 通知](https://github.com/openvinotoolkit/openvino/tree/master/licensing)。wheel 只是 SDK/runtime 的一种来源，不免除这些义务，也不使 Python 成为 CLI 运行依赖。GPU/NPU plugin 依赖的系统驱动及其他组件须按各自条款另行核对。 |
| OpenCL ICD loader 与厂商 GPU runtime/driver | `anr-opencl` 通过 `libloading` 动态加载系统 OpenCL；不内置 ICD loader 或厂商实现。[Khronos OpenCL ICD Loader](https://github.com/KhronosGroup/OpenCL-ICD-Loader) 的 [Apache-2.0 许可](https://github.com/KhronosGroup/OpenCL-ICD-Loader/blob/main/LICENSE) 只覆盖该实现，不能据此推断本机 `OpenCL.dll` / `libOpenCL.so` 的实际来源或厂商 GPU ICD 的许可。具体驱动/runtime 按供应商条款处理，例如 [NVIDIA 驱动许可](https://www.nvidia.com/en-us/drivers/nvidia-license/)；loader 本身不提供硬件 GPU 执行。 |

私有 `anr_tensorrt_native` / `anr_openvino_native` shared shim 的本项目源码仍采用 MIT OR Apache-2.0，但分发 shim 不会替代其 vendor runtime 的分发审查；Rust `libloading`、构建依赖 `cc` 及传递依赖也保留各自通知。SDK headers、链接库、运行库、frontend/plugin 和驱动是不同组件，不应把整个 SDK 目录无差别放入安装包。默认 Release ZIP 仍仅携带 CPU ORT 及既有资产；自行提供原生后端包时，须收集真实依赖闭包及每项适用许可，不将下载/编译成功视为取得再分发授权。

## Rust ORT 绑定与 ONNX Runtime 是两个组件

- Rust `ort` / `ort-sys` 固定为 **2.0.0-rc.15**，声明 **MIT OR Apache-2.0**，来自 [pykeio/ort](https://github.com/pykeio/ort)。其 [ort 源码](https://docs.rs/crate/ort/2.0.0-rc.15/source/) 与 [ort-sys 源码](https://docs.rs/crate/ort-sys/2.0.0-rc.15/source/) 含独立版权和许可。
- Microsoft **ONNX Runtime** 是实际执行推理的原生运行库，不是 Rust 绑定的另一名称；其 [上游源码](https://github.com/microsoft/onnxruntime)、[MIT 许可](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) 和 [ThirdPartyNotices](https://github.com/microsoft/onnxruntime/blob/main/ThirdPartyNotices.txt) 需独立处理。Rust 绑定的默认预编译分发为 1.31.0；GitHub 应用 ZIP 改用 API 27 兼容的 CPU 1.28.0 共享库，Windows/Linux/macOS arm64 来自固定摘要的官方包，Intel macOS 来自固定提交的源码构建。ZIP 的 `licenses/onnxruntime/` 保留**实际分发版本**的 `LICENSE` 与 `ThirdPartyNotices.txt`，不以源仓库当前版本的文件替代。默认分发、自建库和 `ORT_DYLIB_PATH` 指定库的组件集合可能不同。
- CUDA、TensorRT、DirectML、CoreML、XNNPACK 等 provider 及其驱动、SDK、系统框架或随附库各有适用条款。选择 Cargo feature 不代表取得其再分发许可，也不表示这些库已被安装。分发者只应携带有权分发的运行库及其通知，不把 Microsoft、NVIDIA、Apple 或其他供应商组件列为 anr contributors 自有代码。

## 模型权重与测试控制图

DFN3 模型内容不打包进 Rust crate；默认 `download-models` 在构建时从 [固定模型 Release](https://github.com/lona-cn/anr/releases/tag/model-dfn3-fe5eb64fa2e4) 下载并校验，`copy-models` 将模型复制到构建产物旁，`cargo install` 可继续使用持久缓存。关闭 defaults 可禁用下载，运行时不会联网获取模型。GitHub 应用 ZIP 明确包含 `models/dfn3.onnx`。维护者已确认有权将 SHA-256 `fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9` 对应的文件按 **MIT OR Apache-2.0** 分发；两份许可、NOTICE 和资产清单随 ZIP 的 `models/` 一起提供，也可从该模型 Release 获取。`copy-models` 只复制模型文件；自行分发构建产物时，应另外携带适用许可及通知。其他权重、训练数据或导出文件的使用与分发权仍按实际来源确认，不能由 [DeepFilterNet 源码仓库](https://github.com/Rikorose/DeepFilterNet) 的双许可推断。

`tests/fixtures` 中本项目自制的微型 ONNX 控制图用于验证输入输出、状态和错误处理，不含 DFN3 权重，也不宣称具有降噪能力。CI 同时使用这些控制图和经 Release 摘要校验的真实模型；完整模型仍不进入 Rust 源码包。
