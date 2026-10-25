# DFN3 推理性能记录

以下是 2026-10-25 在 Windows 本地使用当前源码完成的一轮全量 `anr-bench` 测量，覆盖工具提供的全部十个后端入口，比较同一 DFN3 模型的连续帧处理耗时。主表已替换为本轮结果；固定参数 shader/kernel 特化的历史对照另节保留，不与本轮混用。数据来自合成输入，不包含音质评估，也不代表其他硬件、驱动或应用负载下的性能。

## 结果

| 后端 | 准备（ms） | 平均每帧（ms） | P99（ms） | FPS | RTF | 超过 10 ms 的帧数 / 1000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gl` | 5263.7 | 1.314 | 2.261 | 760.9 | 0.1314 | 0 |
| `vk` | 1322.9 | 0.400 | 0.736 | 2500.5 | 0.0400 | 0 |
| `dx` | 10153.5 | 0.686 | 1.262 | 1456.5 | 0.0687 | 0 |
| `cuda` | 626.7 | 0.828 | 1.406 | 1208.1 | 0.0828 | 0 |
| `tensorrt` | 18962.2 | 0.432 | 1.104 | 2313.3 | 0.0432 | 0 |
| `openvino` (CPU) | 817.2 | 0.762 | 2.069 | 1311.3 | 0.0763 | 0 |
| `opencl` | 189.7 | 0.759 | 1.783 | 1317.5 | 0.0759 | 0 |
| `ort-cuda` | 1347.9 | 4.550 | 8.180 | 219.7 | 0.4551 | 0 |
| `ort-tensorrt` | 18625.4 | 2.160 | 3.652 | 463.0 | 0.2160 | 0 |
| `ort-directml` | 1885.4 | 1.080 | 1.691 | 926.0 | 0.1080 | 0 |

十项均完成预热与正式测量，JSON 中的 `status` 均为 `ok`，进程退出码为 0。`ort-tensorrt` 初始化期间仍有解析错误诊断，详见后端范围说明；完成测量不代表没有诊断或全程使用 GPU。

### 当前工具的计时定义

- **准备时间**：从开始加载后端到构造函数返回的墙钟时间，包括模型读取、后端初始化，以及在构造期间完成的模型校验、内存改图、设备内存分配、shader/kernel 编译或 engine 构建。后端内部的预执行也在此范围内，例如原生 TensorRT 为捕获 CUDA Graph 而执行的预热。它不包含 Cargo 编译、依赖下载、输入生成、强度设置或 benchmark 额外运行的 100 帧预热；构造函数返回后才发生的延迟初始化不属于该列。
- **平均每帧 / P99**：对每次同步 `NoiseReduction::run` 调用单独计时，包含主机端处理、后端执行、主机/设备传输、完成等待及适用的 worker 通信，不是 GPU kernel 时间。平均值取 1000 次调用耗时的算术平均；分位数采用 nearest-rank：升序排列后取第 `ceil(N × p / 100)` 项，序号从 1 开始，因此这里的 P99 是第 990 项，不做插值。
- **FPS / RTF**：用正式测量循环的总墙钟时间计算，包含循环中的计时与记录开销，而非仅用单次调用耗时之和。`FPS = 帧数 / 总秒数`，`RTF = 总秒数 / 音频秒数`。每帧为 480 / 48000 = 0.01 秒音频，因此未舍入时 `RTF = 100 / FPS`；表中各列分别舍入，FPS 不必等于 `1000 / 平均每帧毫秒数`。
- **超过 10 ms 的帧数**：仅统计单次 `run` 耗时严格大于 10 ms 的调用，恰好 10 ms 不计入。RTF 小于 1 只说明这轮循环的平均处理速度快于输入速率；零超限帧或 P99 小于 10 ms 也不能保证实时应用不丢帧。这不是音频设备到输出的端到端测试：未测设备 I/O、解码、重采样与应用缓冲，`run` 耗时也不等于模型算法延迟。

## 本次测试条件

- 测量时间：2026-10-25 11:27:50–11:29:08（UTC+8）；源码 commit：`2dc8f04afa1db2c8491953f0224ebef2f66c1ce2`，测量时工作区无未提交改动。
- DFN3：48 kHz 单声道，480 采样/帧，降噪强度 80；每个后端预热 100 帧，测量 1000 帧，即正式测量 10 秒音频。
- 输入：一秒长的 220 Hz 和 880 Hz 音调加噪声，固定种子 `0x12345678`，随后循环使用；预热与正式测量连续处理，保留帧间状态和输入位置，不在正式测量前 reset。
- 单轮测试，后端串行运行，顺序同表；复用已有构建目录，未清理后台程序或驱动缓存。结果不包含多轮方差或置信区间，准备时间也不是统一清空缓存后的冷启动测量，不应与旧驱动下的历史结果直接归因比较。
- Windows x64（内部版本 26300）；AMD Ryzen 7 5800X；NVIDIA GeForce RTX 2080 Ti（驱动报告显存 22,528 MiB）。
- Rust 1.99.0（`--release`，debug assertions 关闭）；NVIDIA 驱动 621.01；CUDA 13.3.73、cuDNN 9.26.1.1、TensorRT 10.16.1.11；OpenVINO 2026.4.0（build 22959）；ONNX Runtime 1.31.0、`ort` 2.0.0-rc.15；DirectML 1.15.4。
- 模型 `models/dfn3.onnx` SHA-256：`fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9`。

## 历史：固定参数 shader/kernel 特化对照

本节保留此前对 `gl`、`gles`、`dx`、`wgpu`、`opencl` 固定参数特化的单独对照，本轮没有重测特化前版本。这些数值不是上方主表的测量结果，也不包含 Vulkan 改动。历史对照使用 Windows x64、RTX 2080 Ti、驱动 616.64 和 Rust 1.99.0 release 构建，模型摘要与上文相同。

通过临时程序直接调用五个库，输入与 `anr-bench::synthetic_input` 一致。每个后端串行运行三轮，每轮重新构造实例，强度 80，预热 100 帧后继续测量 1000 次单帧同步 `run`，不 reset。下表取三轮平均每帧耗时的中位数；降幅以未舍入数值计算。OpenCL 一行使用预处理分支修正后的单独三轮复测，其余条目使用此前的五后端对照。未隔离其他系统负载或清空驱动缓存，没有置信区间，不能据此保证其他设备或实时音频性能。GLES/wgpu 仍没有 `anr-bench` 命令行入口。

| 后端 | 特化前平均每帧（ms） | 特化后平均每帧（ms） | 耗时下降 |
| --- | ---: | ---: | ---: |
| `gl` | 1.477 | 1.362 | 7.8% |
| `gles` | 1.486 | 1.407 | 5.3% |
| `dx` | 0.970 | 0.724 | 25.3% |
| `wgpu` | 1.953 | 1.683 | 13.8% |
| `opencl` | 1.019 | 0.706 | 30.7% |

此方案在初始化时为固定模型的 55 条 dispatch 分别编译程序，将算子类型、尺寸、步长、padding、分组和 buffer 偏移作为编译常量，保留衰减强度的动态读取。DXBC 反汇编检查确认 54 条不再声明参数 constant buffer，只有频谱衰减一步保留它。没有强制展开长循环，也没有启用低精度或 fast-math。

初始化成本不可忽略：三轮初始化时间的中位数，DX 从约 0.972 秒增加到 12.228 秒，GL 从 0.123 秒增加到 0.224 秒，GLES 从 0.117 秒增加到 0.232 秒。第一次特化运行的 GL/GLES 分别约 5.68/4.97 秒，后续运行受驱动缓存影响；不是统一冷启动对照。wgpu 的后两轮约 0.39–0.43 秒，首轮约 2.64 秒。应复用已经创建的流实例，不应将本节稳态收益理解为短文件或频繁重建实例也一定更快。

五个后端的现有独立 ONNX reference 回归均通过，包含四种强度、尾部 13 个采样、无效调用、线程移交和 reset；最大样本误差不超过 `2.65e-7`，RMS 不超过 `1.82e-8`，未放宽原有误差门槛。WASM 目标另外通过编译检查，浏览器运行性能未测。

## 当前工具的后端范围

`tensorrt` 是独立的原生实现：两个 TensorRT 神经网络 engine 与 CUDA DSP/FFT 组成整帧执行路径，不经过 ORT。`ort-tensorrt` 则由 ONNX Runtime 的 TensorRT execution provider 执行，并允许 ORT 将不支持或按其分配规则更适合 CPU 的算子放到 CPU。它只显式注册 TensorRT provider，不另加 CUDA provider；provider 注册失败仍是错误，不会重试整个 CPU Session。本轮该项初始化期间输出 `/MatMul_FusedMatMulAndScale` 的 reshape/shape 解析错误：无法将 `[481,1,2]` reshape 为 `[1,2]`；随后仍完成全部预热与测量，报告 `status: ok`。本轮未采集实际 CPU 算子清单，不能将该行解读为全 GPU 执行。

三个 `ort-*` 条目的耗时都包含适用的 CPU 算子及设备间传输。它们共用一个进程级 ORT Environment；因此后续 ORT 条目的准备时间不再包含首次创建该 Environment 的开销。Session 使用单线程算子执行、顺序执行和独立线程池，关闭线程池 spinning；DirectML 另关闭 memory pattern。这些配置是本工具的测量条件，不是对其他 ORT 应用的默认性能判断。

`openvino` 行明确选择 CPU，运行于 Ryzen 7 5800X，JSON 中的 `execution_device` 也为 `CPU`；它不是 OpenVINO GPU 结果，也不是 GPU 失败后的 CPU 回退。本轮未测 OpenVINO GPU/NPU。当前工具有 `--openvino-device CPU|GPU|GPU.n|NPU` 入口，但设备插件、硬件与模型算子仍须兼容，CPU 结果不证明 GPU/NPU 可运行。

当前 `anr-bench` 的 `--backend` 枚举仅包含主表中的十项，本轮已全部测量。GLES、wgpu、ORT CPU、CoreML 和 XNNPACK 没有本工具的测量入口，本轮未测；GLES/wgpu 的历史数据仅见上节。这不等于库或 CLI 不支持这些后端。工具始终编入 `gl`、`vk`、`dx`、`cuda`、`opencl`，默认只选择 `gl,vk,dx`。原生 TensorRT、OpenVINO 和三个 ORT provider 需要各自的 feature。

## 安装工具

`0.3.0` 发布到 crates.io 后，可独立安装 `anr-bench`，默认使用优化构建，不自动下载模型或 ONNX Runtime：

```text
cargo install anr-bench --version 0.3.0 --locked
anr-bench --model /absolute/path/to/dfn3.onnx --backend vk --warmup 100 --frames 1000 --strength 80 --json
```

将模型路径换为自己的文件；此例需要 Windows/Linux 的硬件 Vulkan。已安装的工具可在任意目录运行，模型相对路径以当前工作目录为准。始终编入的后端、可选 features、自备运行库及 SDK 配置见 [独立使用指南](https://crates.io/crates/anr-bench/0.3.0)。下方是本轮源码 checkout 的完整十后端复现流程，不是默认安装就能运行的配置。

## 复现

### 前置条件

完整十项命令面向 Windows x64，`dx` 和 `ort-directml` 需要 Windows。准备最新 Rust stable、Windows SDK/MSVC C++17 工具链，以及支持所选 API 的硬件驱动：

- `gl` 需要 OpenGL 4.3 Core Compute，`vk` 需要 Vulkan 1.1 loader 和硬件设备，`dx` 需要支持 D3D12 Compute 的硬件 adapter；`opencl` 需要 OpenCL 1.2 硬件 GPU ICD 和编译器，仅有 ICD loader 不够。
- `cuda` 运行时需要 NVIDIA Driver API 12.0+ 和 NVRTC 12 或 13；驱动必须兼容实际 NVRTC 生成的 PTX。原生 `tensorrt` 还需 TensorRT 10、CUDA 的头文件与链接库。用 `TENSORRT_ROOT`、`CUDA_ROOT`（或 `CUDA_PATH`）指定 SDK；必要时以 `TENSORRT_LIB_DIR`、`CUDA_LIB_DIR` 覆盖链接库目录。
- 原生 `openvino` 编译需要 OpenVINO C++ SDK 的 headers 和 core 链接库，不需要 ONNX frontend；运行时 `Core::read_model` 需要 ONNX frontend 和所选设备 plugin（此处为 CPU plugin）。用 `OPENVINO_ROOT` 指定 SDK 根目录，或用 `OpenVINO_DIR` 指向其 CMake 目录；官方 wheel 中含 `include/`、`libs/` 的 `openvino` 目录也可作为 SDK。
- ORT provider features 不安装 GPU 驱动、CUDA、cuDNN 或 TensorRT。以下命令通过 `ort-download-binaries` 下载匹配的 ONNX Runtime 分发包，仍须提供相容的厂商运行库。core runtime 与 provider DLL 必须来自同一分发版本，不要混用其他构建目录遗留的 DLL。
- TensorRT/OpenVINO 的构建脚本编译私有 shared shim；本地 Cargo 构建可以从记录的 `OUT_DIR` 加载。厂商运行库及其依赖必须可被加载，CUDA NVRTC/builtins 和 ORT GPU 依赖需位于适用的 DLL 搜索路径。独立分发时还要部署 shim、OpenVINO frontend/plugin 和相应运行库；详见 [库配置与部署](crates/anr/README.md) 和 [ORT 配置](crates/anr-ort/README.md)。

`anr-bench` 不自动下载模型。先准备 `models/dfn3.onnx`，核对上面的 SHA-256；模型来源见 [模型说明](.github/models/NOTICE.md)。以下路径为本机实测配置，其他机器须替换为自己的 SDK/运行库安装目录，再从仓库根目录运行。构建与执行分开，Cargo 编译不计入准备时间：

```powershell
$env:TENSORRT_ROOT = 'F:\Program Files\TensorRT-10.16.1.11'
$env:CUDA_ROOT = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
$env:OPENVINO_ROOT = 'E:\cache\anr-native-sdk\Lib\site-packages\openvino'
$env:PATH = @(
  "$env:CUDA_ROOT\bin\x64"
  "$env:CUDA_ROOT\bin"
  "$env:TENSORRT_ROOT\bin"
  "$env:OPENVINO_ROOT\libs"
  'C:\Program Files\NVIDIA\CUDNN\v9.26\bin\13.4\x64'
  $env:PATH
) -join ';'

cargo build --locked --release `
  --target-dir target/benchmark-ort-gpu `
  -p anr-bench `
  --features "tensorrt,openvino,ort-download-binaries,ort-cuda,ort-tensorrt,ort-directml"

& .\target\benchmark-ort-gpu\release\anr-bench.exe `
  --model models/dfn3.onnx `
  --backend "gl,vk,dx,cuda,tensorrt,openvino,opencl,ort-cuda,ort-tensorrt,ort-directml" `
  --openvino-device CPU `
  --warmup 100 --frames 1000 --strength 80 --json
```

`--json` 将结果写入 stdout，进度和警告写入 stderr，可将 stdout 单独保存为 JSON 文件。报告包含请求配置、采样率、帧长、输入种子、OS/架构、debug-assertions 状态，以及各后端的 `status`、准备时间、实际强度和统计值。OpenVINO 成功结果另含 `execution_device`。`cpu_fallback_allowed` 对 ORT 为 `true`、对原生后端为 `false`，即使显式选择 OpenVINO CPU 也为 `false`；该字段描述允许的分配策略，不统计实际 CPU 算子数量，也不证明 GPU-only 执行。

工具会记录单个后端的错误并继续后续条目；任一后端失败，最终退出码为 1。保存新结果时应同时记录模型 hash、硬件、驱动和运行库版本：JSON 本身不采集这些信息。

若自备 ORT 运行库，将 `ort-download-binaries` 替换为 `ort-load-dynamic`，启动前把 `ORT_DYLIB_PATH` 设为兼容 API 27 的共享库**文件**路径；该分发必须包含请求的 provider。使用单独的 target 目录可避免复用其他 ORT 分发的 provider DLL。

测量与报告定义见 [`benchmark.rs`](crates/anr-bench/src/benchmark.rs)、[`statistics.rs`](crates/anr-bench/src/statistics.rs) 和 [`report.rs`](crates/anr-bench/src/report.rs)；可选后端和 feature 以 [`Cargo.toml`](crates/anr-bench/Cargo.toml) 为准。