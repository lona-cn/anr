# 变更记录

版本按工作区版本号记录；未发布条目不属于下方已有版本。应用包发布流程见 [RELEASING.md](RELEASING.md)。

## Unreleased / 未发布

### 模型准备与分发

- `anr` 和 CLI 默认启用 `download-models`、`copy-models`：构建时从固定 GitHub 模型资产下载 DFN3，校验 SHA-256 并持久缓存；模型复制与下载可独立配置，`ANR_MODEL_PATH` 可指定本地来源。关闭默认 features 可去掉模型下载器依赖。
- 新增 `anr::default_model_path()`，先查找可执行文件旁的 `models/dfn3.onnx`，再查找构建时记录的源路径。CLI 的 `--model` 改为可选，显式路径优先；自动选择的模型同样受输出覆盖保护。raw 示例也支持省略模型路径。运行时不下载模型，Rust 源码包不嵌入模型内容。

### 原生推理后端

- 新增独立于 ORT 的 `anr-cuda`、`anr-tensorrt`、`anr-openvino`、`anr-opencl`，以及对应 facade feature/module 和 CLI `--backend` 选项。CLI 默认后端与应用 ZIP 仍使用 CPU ORT；`--provider` 仍只适用于 ORT。
- CUDA 使用 Driver API/NVRTC 与 CUDA Graph；OpenCL 使用 OpenCL 1.2 硬件 GPU。两者完整执行固定 DFN3 的 FFT、特征处理、网络、频谱滤波和音频合成，状态常驻 GPU，不回退 CPU。
- TensorRT 使用标准层神经网络 engine，并与原生 CUDA DSP/FFT 组合为逐帧 graph；采用 FP32，禁用 TF32，不依赖 ORT 或自定义推理 plugin。
- OpenVINO 直接导入摘要校验后的原 ONNX，以 stateful `InferRequest` 保留跨帧状态。新增 `--openvino-device CPU|GPU|GPU.n|NPU`，默认明确选择 CPU；设备或算子不支持时返回错误，不改用 CPU。NPU 可能使用低精度，不能等同于 FP32 执行。
- TensorRT/OpenVINO 构建需要厂商 C++ SDK，部署需附带私有共享 shim 与适用运行库；加载失败作为后端错误返回。SDK-free CI 排除这两个包，SDK 配置及平台限制见 [库文档](crates/anr/README.md)。

### ONNX Runtime

- ORT 默认策略允许按算子支持情况和 ORT 分配规则使用 CPU kernel，CLI 与 benchmark 不再要求全算子 GPU。自定义 Session 保留调用方设置；请求的 provider 注册或初始化失败仍报错。
- CUDA、TensorRT、DirectML 加载入口在内存中准备模型，保持权重、状态布局及动态衰减语义，不改写原 ONNX。CUDA 展开 sigmoid `FusedConv`，TensorRT/DirectML 另展开 ReLU 融合卷积和符合条件的带缩放矩阵乘法；准备只发生在初始化阶段，不生成磁盘模型缓存。
- 三个 GPU 目标共用静态状态序列和安全张量分支转换，将仅用于全零判断的 `float32` `NonZero` 改为固定形状归约，清理恒定布尔选择及无状态死计算；未知或有状态计算保留执行依赖。补充广播、子图捕获、可覆盖常量、非有限值及旧 opset 类型兼容回归。
- 新增并导出 `session_from_file`、`prepare_model`、`ModelTarget`，供库、CLI 和 benchmark 共用，保留调用方 builder 配置及显式外部权重目录。CPU 和其他不需要准备的 provider 继续使用文件加载路径。
- 使用完整 `onnx-protobuf` 消息保留未知字段，并补充其 MPL-2.0 源码获取说明。

### 浏览器 WebGPU

- `anr-wgpu` 新增 `wasm32-unknown-unknown` 支持，通过异步 `anr::wgpu::web::DFN3` 或生成的 JavaScript `DFN3` 类加载模型字节并处理音频；共用固定模型校验、WGSL 与调度，原生同步 API 不变，不提供 CPU/WebGL fallback。
- 新增 Web 包构建脚本、支持缓冲转移的串行 module Worker 示例与浏览器 reference 回归。异步接口拒绝重叠处理和处理期间的状态修改，并处理初始化、映射失败及取消时的资源清理。

### ANR-Lite-48

- 新增独立的 `anr-lite`，使用 Burn 0.22.1 实现 ANR-Lite-48：64-band ERB/96-bin 低频双分辨率编码器、两层 128 维四分组 GRU 与跨组混合、全带增益和五 tap 复数残差输出。默认 CPU Flex，可选显式 wgpu 独显，不改变 DFN3 后端或 CLI。
- 提供可微 960 点混合基 FFT、稀疏 ERB 特征、因果归一化、显式流式状态、overlap-add、尾帧补零与 flush；整段入口补偿 1440 samples 的流式延迟。支持批处理、跨帧梯度、显式 detach 和 Burn Module records，并提供合成音频示例。不附带预训练权重，不自动下载数据集。
- 新增可选 `train` feature 与 `anr-lite-train`：使用 VoiceBank-DEMAND 28-speaker 配对数据、AdamW 和压缩幅度谱/复数谱/波形损失训练，支持 CPU Flex 和显式 wgpu 独显。
- 训练严格检查同名等长的 48 kHz 单声道 PCM16 配对，按说话人划分验证集，对短片段补零使用有效长度掩码；仅用验证集选模，官方测试集需显式启用。每次在新目录保存 best/last 模型和训练记录；checkpoint 包含模型及归一化状态，不包含精确续训所需的优化器/RNG 状态。
- 验证和可选训练后测试改为每批最多八段完整语句推理；计算损失前去除各自补零，保留逐句损失及按有效采样数加权的归约，不改变训练目标或评估样本集。
- 新增 `anr-lite-evaluate`：加载 Burn checkpoint，批量处理不同长度语句，补偿延迟并输出原始长度的 48 kHz 单声道 float32 WAV；不剪裁或调整音量，GPU 需显式选择。
- 新增 VoiceBank-DEMAND 客观评估工具：独立校准并补偿 DFN3 固定延迟，比较完整 824 个官方测试样本的 WB-PESQ、STOI、SI-SDR、SNR，保存逐文件结果、配对差值和 bootstrap 置信区间。输入必须同名、等长且有效，失败样本不静默省略；数量限制仅供短跑检查。使用方法与评价边界见 [ANR-Lite 文档](crates/anr-lite/README.md)。

### 性能测量

- 新增工作区工具 `anr-bench`：使用固定种子的连续 48 kHz 单声道输入，串行比较后端，初始化单独计时，支持预热、后端/强度选择及 JSON。报告同步调用延迟、nearest-rank P50/P95/P99、FPS、RTF 与超过 10 ms 的帧数；单后端失败后继续运行，最终返回非零退出码。不包含在应用 ZIP 中。
- 默认构建不依赖 ORT，编入 `gl`、`vk`、`dx`、`cuda`、`opencl`，默认测量列表为 `gl,vk,dx`；TensorRT/OpenVINO 通过可选 SDK features 编入。另支持 `ort-cuda`、`ort-tensorrt`、`ort-directml` 与下载/动态运行库选项，ORT 后端共用一个 Environment。
- JSON 记录 ORT 的 CPU 算子执行策略及 OpenVINO 请求/实际执行设备。`cpu_fallback_allowed` 表示策略，不是 GPU 算子占比；显式选择 OpenVINO CPU 仍标为 false。测量包含同步执行、CPU 算子和设备传输。历史数据、硬件范围及复现方法见 [benchmark.md](benchmark.md)。

## 0.1.1

### 原生 Compute

- 新增固定模型的 `anr-gl`、`anr-vk`、`anr-dx`、`anr-wgpu` 后端，不依赖 ONNX Runtime；原 ONNX 作为 SHA-256 校验的参数来源。DX 使用原生 D3D12，不是 DirectML；VK/DX 不经过 wgpu。
- 新增 `anr-gles`，支持 Windows WGL ES profile 与 Linux 硬件 EGL pbuffer，要求原生 OpenGL ES 3.1+ context，不退回桌面 GL、ORT 或 CPU；未提供 Android/WebGL/macOS context。
- 各后端通过 `anr-dfn3` 共用权重布局、固定执行顺序、工作线程与 WGSL，使用 Naga 生成 GLSL、GLSL ES、SPIR-V 或 HLSL。完整 GPU 流程包含 960 点混合基 FFT、归一化、卷积、分组线性层、GRU、频谱滤波与 overlap-add，权重和双状态缓冲常驻 GPU。
- GLES 在 fence 后映射当前输出帧并检查解除映射结果，不使用桌面专用 `glGetBufferSubData`；按 FFT 实际需求检查 15,360 字节共享内存，修正原先错误的 32 KiB 要求。

### 接口与持续集成

- facade 新增 `gl`、`gles`、`vk`、`wgpu`、`dx` features 和对应 `DFN3` 模块；CLI 新增 `--backend ort|gl|gles|vk|wgpu|dx`，`--provider` 仅适用于 ORT。默认构建和应用 ZIP 仍使用 ORT。
- 增加独立 ONNX 连续帧数值回归，以及强度、尾帧、无效输入、线程移交、reset 和多声道/批处理覆盖。
- CI 将普通工作区测试、ORT 真实模型测试及 Compute-only CLI 的非 GPU 检查分开，不在无 GPU runner 上运行 ignored 硬件测试。

## 0.1.0

初始版本包含公共接口、ORT 实现、facade 与独立 CLI。

### Rust 库

- `anr-core` 提供不依赖推理运行库的 `NoiseReduction: Send`、`Error`、`Result`；`anr` 为统一入口，默认不启用执行后端，提供 ORT、动态加载及 CUDA/TensorRT/DirectML/CoreML/XNNPACK provider 编译选项。
- `anr-ort` 提供有状态 DFN3，支持模型文件或自定义 Session。构造时检查具名 `f32` 三入三出的维数及已知静态长度，兼容一维符号维度，以及衰减量和 `lsnr` 的标量形式。
- DFN3 处理 48 kHz 单声道音频，每帧 480 个采样、45,304 个状态值；尾帧补零并返回原始长度，`reset` 保留强度。强度 0–100 映射到 0–60 dB 衰减上限；忽略 NaN，钳制无穷和越界值。
- 固定使用 `ort 2.0.0-rc.15` 与 API 27，改进失败加载后的初始化行为；默认预编译运行库为 1.31.0，也支持兼容 API 27 的自备运行库（包括 1.28.0），不启用 `ndarray`。仅支持最新 Rust stable，不维护固定 MSRV。
- 动态库提供可失败的显式加载入口；`DFN3::from_file(&Environment, path)` 由应用持有环境，多个实例可共享。CPU 构造显式选择内置 provider，不继承自动 NPU 策略，也不强制 minimal runtime 提供设备发现。
- 拆分 `ort` 与 `ort-download-binaries`；保留默认 CPU 运行库下载，关闭默认 features 的动态模式不再引入下载器和 TLS/OpenSSL 构建依赖。
- 提供 raw f32le 文件处理示例、独立 crate 文档和 MIT OR Apache-2.0 许可文本。强度边界回归采用容差以适配 ARM64 浮点归约，不改变推理计算。

### 命令行工具

- 新增 `anr-cli`，二进制名为 `anr`，支持文件降噪、串行批处理、信息查询及 raw f32le stdin/stdout。默认 CPU ORT，provider 显式选择，注册失败不静默回退。
- 使用 Symphonia 解码、rubato 带限流式重采样、每声道独立模型状态和跨解码包模型分帧；支持 PCM16/PCM24/float32 WAV 或 raw 输出，默认保留采样率、声道及帧数，不推测模型延迟补偿。
- 默认不覆盖目标，输出完成并 finalize 后才提交同目录临时文件；单文件失败后批次继续，保留已成功结果。二进制 stdout 与诊断 stderr 分离，支持取消、下游 BrokenPipe，并在 JSON 错误中保留 BrokenPipe 分类。
- 以实际 FileId 保护输入、模型与本批已完成结果，写入前和提交前均复核；已保存但身份记录失败时报告具体状态并停止后续处理。
- WAV EOF 校验对固定宽度 codec 使用可靠声明帧数与实际解码帧数，拒绝无 payload 或尾部截断；未知长度标记不强行生成长度预算。

### CI 与发布

- 配置 Windows x64、Linux x64、macOS arm64/x64 原生 Rust stable 检查，包含固定模型校验、真实模型、无后端模式、feature 编译、Clippy、文档及解压应用 ZIP 后的推理。Cargo 缓存区分平台、编译器和工作流，并缓存 ORT 下载及 Intel macOS CPU 运行库/许可。
- CI 在 main/master push、版本 tag、pull request 和手动触发时运行。仅取消同一 PR 的旧任务，普通 push 按提交运行。
- 固定模型资产单独分发 `dfn3.onnx`、模型许可、NOTICE、清单与 SHA256SUMS；该文件由维护者确认可按 MIT OR Apache-2.0 分发。CLI 第三方说明补充 Symphonia 的 MPL-2.0 义务以及 ORT、模型的许可边界。
- main/master push 在四平台通过后创建不可变提交预发布；`v<工作区版本>` tag 发布对应版本并拒绝版本错配。PR 和手动运行不发布，工作流不发布 crates.io 包。
- 应用 ZIP 包含 CLI、CPU ONNX Runtime 1.28.0、DFN3 模型及所需许可，使用动态加载，无需用户配置运行库搜索路径。**本版本的 `--model` 仍为必填参数**；自动模型准备属于未发布变更。
- 先上传 draft，核对所有 ZIP 与 SHA256SUMS 后再公开；不移动 tag，不覆盖公开版本或同名异摘要资产。发布只使用内置 `GITHUB_TOKEN`，仅发布 job 获得 Contents 写权限；额外 workflow 授权不足时明确失败，不替换目标提交。
- 发布前按完整 Git ref 检查 tag，兼容 lightweight/annotated tag 并核对提交，修正首次预发布因不存在 ref 的 `422` 响应而失败的问题。
