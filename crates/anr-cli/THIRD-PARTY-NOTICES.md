# 第三方组件与分发说明

`anr-cli` 自有代码采用 MIT OR Apache-2.0。依赖、原生运行库、驱动和模型各有自己的许可；分发时按实际 `Cargo.lock`、features 和随附二进制收集版权、许可及适用 NOTICE。本说明列出主要组件与源码入口，不是所有 feature 组合的完整许可清单。

## 音频解码：Symphonia 0.5.5（MPL-2.0）

CLI 使用纯 Rust 的 Symphonia 解码，启用 `all`，不依赖系统 FFmpeg。其源码通知为 **Copyright (c) 2019–2022 The Project Symphonia Developers**；其他文件中的原始通知也应保留。

- 上游源码与原始版权通知：[pdeljanov/Symphonia](https://github.com/pdeljanov/Symphonia)。
- 完整许可：[Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/)。
- 精确版本源码入口：[symphonia 0.5.5 源码](https://docs.rs/crate/symphonia/0.5.5/source/)，或下载 [crates.io 源码包](https://crates.io/api/v1/crates/symphonia/0.5.5/download)。

同版本组件的源码分别位于以下入口；不是只提供顶层 façade 就足以覆盖静态链接进去的 codec：

- [symphonia-core](https://docs.rs/crate/symphonia-core/0.5.5/source/)、[symphonia-metadata](https://docs.rs/crate/symphonia-metadata/0.5.5/source/)、[symphonia-utils-xiph](https://docs.rs/crate/symphonia-utils-xiph/0.5.5/source/)。
- [symphonia-bundle-flac](https://docs.rs/crate/symphonia-bundle-flac/0.5.5/source/)、[symphonia-bundle-mp3](https://docs.rs/crate/symphonia-bundle-mp3/0.5.5/source/)。
- [symphonia-codec-aac](https://docs.rs/crate/symphonia-codec-aac/0.5.5/source/)、[symphonia-codec-adpcm](https://docs.rs/crate/symphonia-codec-adpcm/0.5.5/source/)、[symphonia-codec-alac](https://docs.rs/crate/symphonia-codec-alac/0.5.5/source/)、[symphonia-codec-pcm](https://docs.rs/crate/symphonia-codec-pcm/0.5.5/source/)、[symphonia-codec-vorbis](https://docs.rs/crate/symphonia-codec-vorbis/0.5.5/source/)。
- [symphonia-format-caf](https://docs.rs/crate/symphonia-format-caf/0.5.5/source/)、[symphonia-format-isomp4](https://docs.rs/crate/symphonia-format-isomp4/0.5.5/source/)、[symphonia-format-mkv](https://docs.rs/crate/symphonia-format-mkv/0.5.5/source/)、[symphonia-format-ogg](https://docs.rs/crate/symphonia-format-ogg/0.5.5/source/)、[symphonia-format-riff](https://docs.rs/crate/symphonia-format-riff/0.5.5/source/)。

[MPL-2.0 第 3.1–3.4 条](https://www.mozilla.org/en-US/MPL/2.0/#responsibilities)规定：

- 被覆盖源码及其修改须继续按 MPL-2.0 提供，告知接收者许可及其获取方式，并保留版权、专利、免责声明等通知的实质内容。
- 分发包含这些组件的 executable 时，须告知接收者如何及时以合理方式取得对应 Source Code Form，收费不超过分发成本。独立文件中的 CLI 代码可以采用其他许可，但不得限制接收者对 MPL 源码的权利。
- 未修改的组件可从上述精确版本入口取得；使用修改版或 patch 时，须提供**实际使用的修改后源码**，只有上游链接不够。分发包应附源码归档或有效下载位置，并提供 MPL 完整文本或明确的许可获取入口。离线归档可用 `cargo vendor --locked <源码归档目录>` 收集锁定依赖，再核对实际 patch 和通知。

软件许可不构成 codec 专利或其他权利在所有地区均已获授权的保证。

## ONNX 消息解析：onnx-protobuf 0.2.3（MPL-2.0）

ORT 适配器使用 `onnx-protobuf` 解析并转换内存中的 ONNX 图。0.2.3 的 package 作者信息为 Aster；原始许可和文件通知以该版本源码为准。

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

传递依赖同样保留各自义务。MIT 的版权及许可文本、Apache-2.0 的许可及适用 NOTICE，均不能用项目自身的版权声明替换；ISC、Zlib 等依赖也须保留其要求的通知。

## 可选 Compute 后端

只启用 `gl`、`gles`、`vk`、`wgpu`、`dx` 不引入 Rust ORT 或 ONNX Runtime。主要绑定、编译器和参数处理依赖如下；链接指向精确版本的源码及其许可文件：

| 组件 | 许可 | 用途 / 源码 |
| --- | --- | --- |
| glow 0.16.0 | MIT OR Apache-2.0 OR Zlib | [GL/GLES 调用](https://docs.rs/crate/glow/0.16.0/source/) |
| glutin 0.32.3 | Apache-2.0 | [原生 GL/GLES context](https://docs.rs/crate/glutin/0.32.3/source/) |
| raw-window-handle 0.6.2 | MIT OR Apache-2.0 OR Zlib | [窗口句柄](https://docs.rs/crate/raw-window-handle/0.6.2/source/) |
| ash 0.38.0+1.3.281 | MIT OR Apache-2.0 | [Vulkan 绑定](https://docs.rs/crate/ash/0.38.0+1.3.281/source/) |
| wgpu / Naga 30.0.1 | MIT OR Apache-2.0 | [wgpu](https://docs.rs/crate/wgpu/30.0.1/source/) 原生/WebGPU 后端；[Naga](https://docs.rs/crate/naga/30.0.1/source/) 将 WGSL 转为 GLSL、GLSL ES、SPIR-V、HLSL |
| windows 0.62.2 | MIT OR Apache-2.0 | [D3D12、DXGI、D3DCompiler 47 绑定](https://docs.rs/crate/windows/0.62.2/source/)；不是 DirectML |
| prost 0.14.4 | Apache-2.0 | [ONNX 参数解析](https://docs.rs/crate/prost/0.14.4/source/) |
| sha2 0.10.9 | MIT OR Apache-2.0 | [模型摘要](https://docs.rs/crate/sha2/0.10.9/source/) |
| bytemuck 1.25.2 | Zlib OR Apache-2.0 OR MIT | [字节视图](https://docs.rs/crate/bytemuck/1.25.2/source/) |

这些 Rust 组件的许可不覆盖系统图形库、驱动或模型。crate 不安装驱动；分发者需按目标平台配置硬件、真实 GL/GLES context 或其他运行环境，并按实际依赖版本保留通知。

## 原生 CUDA、TensorRT、OpenVINO 与 OpenCL

`cuda`、`tensorrt`、`openvino`、`opencl` 是独立于 ORT 的可选后端；它们不启用 Rust `ort` 或 ONNX Runtime。`ort-cuda`、`ort-tensorrt` 则仍属于 ORT provider 链。以下边界适用于实际启用和分发的组件，不意味着所有库都在默认 CPU ORT ZIP 中：

| 组件 | 许可边界与一手来源 |
| --- | --- |
| `cudarc` 0.19.10 | 原生 CUDA 使用的 Rust Driver/NVRTC 动态绑定，**MIT OR Apache-2.0**；[上游](https://github.com/chelsea0x3b/cudarc)、[MIT](https://github.com/chelsea0x3b/cudarc/blob/main/LICENSE-MIT)、[Apache-2.0](https://github.com/chelsea0x3b/cudarc/blob/main/LICENSE-APACHE)、[精确版本源码](https://docs.rs/crate/cudarc/0.19.10/source/)。绑定许可不覆盖 NVIDIA 库。 |
| NVIDIA CUDA Driver、NVRTC / NVRTC builtins、CUDA Runtime | 本项目的 CUDA kernel 源码不改变 NVIDIA 原生组件的许可。CUDA 后端运行时加载 Driver 和 NVRTC；TensorRT shim 还链接 CUDA Runtime 等库。按实际 SDK/运行库版本核对 [CUDA Toolkit EULA、Supplement 与 Attachment A](https://docs.nvidia.com/cuda/eula/index.html)，只分发被明确允许且满足分发条件的文件，保留适用许可和通知。驱动应通过厂商支持的安装渠道提供；不能由 Rust 绑定双许可推断驱动、整个 Toolkit 或任意 DLL 可自由再分发。 |
| NVIDIA TensorRT | `anr-tensorrt` 编译本项目私有 C ABI shim，神经网络使用 TensorRT standard layers，DSP 使用 CUDA；不使用 ORT、ONNX parser 或自定义推理 plugin，但实际 `nvinfer` 等 vendor runtime 仍受 [TensorRT SDK License Agreement 及 Supplement](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html) 和随 SDK 的通知约束。[TensorRT 开源仓库的 Apache-2.0 许可](https://github.com/NVIDIA/TensorRT/blob/main/LICENSE) 不可泛化为所有二进制 SDK 的许可。 |
| OpenVINO Runtime、ONNX frontend、设备 plugins | `anr-openvino` 的私有 shim 调用原生 C++ Runtime。OpenVINO 项目采用 **Apache-2.0**，见 [上游 LICENSE](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE)；分发时保留实际所用版本的许可、版权、适用 NOTICE 和 [third-party-programs 通知](https://github.com/openvinotoolkit/openvino/tree/master/licensing)。wheel 只是 SDK/runtime 的一种来源，不免除这些义务，也不使 Python 成为 CLI 运行依赖。GPU/NPU plugin 依赖的系统驱动及其他组件须按各自条款另行核对。 |
| OpenCL ICD loader 与厂商 GPU runtime/driver | `anr-opencl` 通过 `libloading` 动态加载系统 OpenCL；不内置 ICD loader 或厂商实现。[Khronos OpenCL ICD Loader](https://github.com/KhronosGroup/OpenCL-ICD-Loader) 的 [Apache-2.0 许可](https://github.com/KhronosGroup/OpenCL-ICD-Loader/blob/main/LICENSE) 只覆盖该实现，不能据此推断本机 `OpenCL.dll` / `libOpenCL.so` 的实际来源或厂商 GPU ICD 的许可。具体驱动/runtime 按供应商条款处理，例如 [NVIDIA 驱动许可](https://www.nvidia.com/en-us/drivers/nvidia-license/)；loader 本身不提供硬件 GPU 执行。 |

私有 `anr_tensorrt_native` / `anr_openvino_native` shared shim 的本项目源码采用 MIT OR Apache-2.0；其 vendor runtime 仍需单独审查。动态加载依赖 `libloading` 采用 ISC，构建依赖 `cc` 采用 MIT OR Apache-2.0，传递依赖按实际版本处理。SDK headers、链接库、运行库、frontend/plugin 和驱动应分别列出，不要整目录复制 SDK。默认应用 ZIP 包含 CPU ORT 和模型，不包含这些原生后端运行库；定制包应收集实际依赖闭包及每项适用许可。

## Rust ORT 绑定与 ONNX Runtime 是两个组件

- Rust `ort` / `ort-sys` 固定为 **2.0.0-rc.15**，声明 **MIT OR Apache-2.0**，来自 [pykeio/ort](https://github.com/pykeio/ort)。其 [ort 源码](https://docs.rs/crate/ort/2.0.0-rc.15/source/) 与 [ort-sys 源码](https://docs.rs/crate/ort-sys/2.0.0-rc.15/source/) 含独立版权和许可。
- Microsoft **ONNX Runtime** 是实际推理的原生运行库；1.28.0 为 MIT，**Copyright (c) Microsoft Corporation**。对应版本的[源码](https://github.com/microsoft/onnxruntime/tree/v1.28.0)、[LICENSE](https://github.com/microsoft/onnxruntime/blob/v1.28.0/LICENSE)、[ThirdPartyNotices](https://github.com/microsoft/onnxruntime/blob/v1.28.0/ThirdPartyNotices.txt) 独立于 Rust 绑定处理。绑定默认下载分发使用 1.31.0；GitHub 应用 ZIP 使用 API 27 兼容的 CPU 1.28.0，Windows/Linux/macOS arm64 来自固定摘要的官方包，Intel macOS 来自固定提交的源码构建。ZIP 的 `licenses/onnxruntime/` 保留**实际分发版本**的通知，不用其他版本替代。自建库或 `ORT_DYLIB_PATH` 指定库应核对自己的组件集合。
- CUDA、TensorRT、DirectML、CoreML、XNNPACK 等 provider 及其驱动、SDK、系统框架或随附库各有适用条款。选择 Cargo feature 不代表取得其再分发许可，也不表示这些库已被安装。分发者只应携带有权分发的运行库及其通知，不把 Microsoft、NVIDIA、Apple 或其他供应商组件列为 anr contributors 自有代码。

## 模型权重与测试控制图

Rust 源码包不包含完整 DFN3 模型。默认 `download-models` 在构建时从[固定模型 Release](https://github.com/lona-cn/anr/releases/tag/model-dfn3-fe5eb64fa2e4) 下载并校验，`copy-models` 只复制模型文件；`cargo install` 可使用持久缓存。关闭 defaults 可禁用下载，运行时不联网获取模型。自行分发构建产物时，须另附模型许可和通知。

GitHub 应用 ZIP 包含 `models/dfn3.onnx`。维护者已确认有权按 **MIT OR Apache-2.0** 分发这份文件，SHA-256 为：

```text
fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9
```

两份模型许可、NOTICE 和资产清单随 ZIP 放在 `models/`，也可从上述模型 Release 获取。该授权仅适用于这份文件，不声明上游作者身份；其他模型、权重、训练数据和导出文件的权利按各自来源确认，不能由 [DeepFilterNet 源码仓库](https://github.com/Rikorose/DeepFilterNet) 的双许可推断。

`tests/fixtures` 中本项目自制的微型 ONNX 控制图用于验证输入输出、状态和错误处理，不含 DFN3 权重，也不宣称具有降噪能力。CI 同时使用这些控制图和经 Release 摘要校验的真实模型；完整模型仍不进入 Rust 源码包。
