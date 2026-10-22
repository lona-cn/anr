# 变更记录

## 未发布

- `anr-lite` 新增可选 `train` feature 与 `anr-lite-train`：读取已解压的 VoiceBank-DEMAND 28-speaker，使用 AdamW 和可微压缩幅度谱／复数谱／波形损失进行真实训练，支持 CPU Flex 与显式 wgpu 独显。
- 训练程序严格检查同名等长的 48 kHz 单声道 PCM16 配对，按说话人划分验证集，掩蔽短片段补零；仅用验证集选模，官方测试集需显式启用。每次使用新目录保存 best/last 模型和训练记录，提供短跑步数／评估数量限制，不修改数据集，不将模型权重 checkpoint 冒充包含优化器状态的精确续训快照。
- `anr-wgpu` 新增 `wasm32-unknown-unknown` 浏览器 WebGPU：通过 `anr::wgpu::web::DFN3` 异步处理，或使用生成的 JavaScript `DFN3` 类；共用模型字节校验、权重布局、WGSL 和固定调度，原生同步 API 不变，不新增 CPU/WebGL fallback。
- 新增 Web 包构建脚本、可转移缓冲的串行 module Worker 示例与真实 WASM 浏览器 reference 回归；覆盖强度/reset、尾帧状态、无效输入、并发拒绝和处理中释放包装对象，并为初始化/映射错误与 future 取消提供设备和状态清理。
- 新增独立的 `anr-lite`，使用 Burn 0.22.1 实现 ANR-Lite-48：64-band ERB／96-bin 低频双分辨率编码器、两层 128 维四分组 GRU 与跨组混合、全带增益和五 tap 复数残差输出。默认使用 CPU Flex，可选 wgpu；不修改现有 DFN3 推理后端或 CLI。
- `anr-lite` 提供可微 960 点混合基 FFT、稀疏 ERB 特征、因果归一化、显式流式状态、overlap-add、尾帧补零与 flush；整段入口补偿 1440 samples 的流式延迟。支持批处理、跨帧梯度、显式 detach 和 Burn Module records。
- 新增未训练模型的合成音频 smoke 示例及数值／梯度回归；不启动训练、不下载数据集、不提供预训练权重，不以随机初始化输出宣称降噪质量或性能优势。
- `anr` 和 CLI 默认启用 `download-models`、`copy-models`：构建时从固定 GitHub Release 下载 DFN3，校验 SHA-256 并持久缓存，独立控制向 profile、examples、tests 构建目录复制模型；可用构建环境变量 `ANR_MODEL_PATH` 选择本地文件，关闭 defaults 则不引入模型下载器依赖。
- 新增 `anr::default_model_path()`，先查找可执行文件旁的 `models/dfn3.onnx`，再使用构建时记录的源路径。CLI 的 `--model` 改为可选，显式路径优先；自动选择的模型同样受输出覆盖保护。raw 示例也支持省略模型路径，运行时不下载模型。
- 新增工作区工具 `anr-bench`，串行比较 `anr-gl`、`anr-vk`、`anr-dx` 的连续帧推理性能；共用固定种子的 48 kHz 单声道输入，初始化单独计时，支持预热、后端选择、降噪强度和 JSON。报告同步调用延迟、nearest-rank P50/P95/P99、FPS、RTF 及超过 10 ms 的帧数；任一后端失败仍继续测试其余后端，并返回非零退出码。默认构建不依赖 ORT，不改动原生推理实现，也不加入应用 Release ZIP。
- `anr-bench` 增加可选的 `ort-cuda`、`ort-tensorrt`、`ort-directml` 后端，以及 `ort-download-binaries` / `ort-load-dynamic` 运行库来源选项；共用一个 ORT Environment，缺失运行库或 provider 作为单后端错误返回，新增 opt-in 硬件回归。
- 统一项目 ORT 默认策略：允许不支持或按 ORT 分配规则更适合 CPU 的算子使用 CPU kernel，CLI 和 benchmark 不再强制全算子 GPU；库保留调用方 Session 设置，provider 注册或初始化失败仍报错。benchmark 用 `cpu_fallback_allowed` 标注执行策略，耗时包含 CPU 算子及设备传输，不冒充纯 GPU kernel 时间。
- 新增 ORT 加载前的运行时内存改图：CUDA 展开不支持的 `FusedConv(Sigmoid)`；TensorRT 另将静态状态序列访问转换为 `Split`，将可以证明安全的固定形状张量分支转换为 `Where`。保持权重、状态布局及动态衰减语义，不改写原 ONNX、不生成派生模型或磁盘缓存、不在每帧重复转换。
- 新增并导出 `session_from_file`、`prepare_model`、`ModelTarget`，CLI 与 benchmark 共用加载入口；保留调用方 builder 配置及显式外部权重目录，注册失败仍报错。CPU、DirectML 和原生 GL/VK/DX 加载不变，不使用 TRT 全局算子黑名单或整 Session CPU 重试。
- 使用完整 `onnx-protobuf` 消息保留未知字段，补充其 MPL-2.0 源码获取说明。新增数值回归覆盖卷积残差、静态及动态索引、分支局部命名、条件形状和不能提前执行的分支；真实模型 GPU 回归覆盖动态强度、reset 和尾帧。
- 新增不依赖 ORT 的 `anr-cuda`、`anr-tensorrt`、`anr-openvino`、`anr-opencl`，以及对应 facade features/modules 和 CLI `--backend`。原有默认 ORT 构建、默认后端和 CPU Release ZIP 不变，`--provider` 仍只适用于 ORT。
- CUDA 使用 Driver API/NVRTC、固定显存与双方向 CUDA Graph；OpenCL 使用硬件 GPU、原生 OpenCL C 1.2 kernels 和 in-order queue。两者完整执行固定 DFN3 的 FFT、特征归一化、网络、频谱滤波及音频合成，不回退 CPU。
- TensorRT 将共享固定调度转换为两个标准层 NN engine，与原生 CUDA DSP/FFT 共同捕获为整帧 Graph；保留 encoder 后的 DF-history 更新，使用 256 字节对齐 scratch 与 GRU 状态绑定，不将完整模型藏在单一 plugin 中。
- OpenVINO 直接导入经摘要校验的原 ONNX，用 MakeStateful 保持每个 InferRequest 的跨帧状态；新增 `--openvino-device CPU|GPU|GPU.n|NPU`，默认明确使用 CPU，设备或算子不支持时不改用 CPU。NPU 的低精度硬件不冒充 FP32 计算。
- `anr-bench` 接入四个原生后端，CUDA/OpenCL 随 native 构建编入，TensorRT/OpenVINO 通过独立可选 SDK features 编入；默认测量列表仍为 `gl,vk,dx`。OpenVINO JSON 记录请求设备与实际执行设备；显式 CPU 推理仍标为不允许 fallback。
- Windows 上四后端均通过固定阈值的独立连续帧数值回归、强度切换、尾帧、无效调用、线程移交和 reset，以及 CLI raw/WAV、独立立体声与批处理测试。CUDA/TensorRT/OpenCL 使用 RTX 2080 Ti，OpenVINO 使用 Ryzen 7 5800X CPU；七后端 benchmark 已实跑。OpenVINO GPU 原图在 NonZero kernel 编译处失败，NPU 与跨平台设备执行未验证，不据此声称全部设备支持或实时保证。
- TensorRT/OpenVINO 需要显式配置厂商 C++ SDK，并分发私有共享 shim 与适用运行库；按需加载失败作为后端错误返回。无 SDK 的 CI 检查明确排除这两个包，保留其他后端与 ORT provider 检查；有 SDK 的配置单独验收。

## 0.1.1

- 新增 `anr-gles`、`gles` feature 和 `--backend gles`：复用 GL 执行器及统一 WGSL，生成 GLSL ES 310，强制原生 ES 3.1+ context，不退回桌面 GL、ORT 或 CPU。Windows 使用 WGL ES profile，Linux 使用硬件 EGL pbuffer；尚未提供 Android/WebGL/macOS context。
- GLES 使用 fence 后映射当前输出帧，不调用桌面专用 `glGetBufferSubData`；检查原生 `glUnmapBuffer` 返回值。共享内存要求按 FFT 的实际 15,360 字节检查，不再错误要求 ES 设备提供 32 KiB。
- GLES 已在 Windows / RTX 2080 Ti 上通过 61 帧独立数值基准、GLES-only 及混合后端 CLI 验收；用户录音输出与原 GL 文件摘要一致。WSL Ubuntu 24.04 的 GLES-only 编译、非 GPU 测试和 Clippy 通过，未宣称 Linux 或移动端 GPU 已验证。
- 新增固定 DFN3 模型的原生 `anr-gl`、`anr-vk`、`anr-dx` 与独立 `anr-wgpu` 后端，均不依赖 ONNX Runtime；原 ONNX 仅作为按 SHA-256 校验的参数容器。DX 为原生 D3D12，不是 DirectML；VK/DX 不经 wgpu。
- 新增 `anr-dfn3` 共用权重布局、固定执行顺序、工作线程和 WGSL 算子，通过 Naga 生成 GLSL、SPIR-V、HLSL；FFT、特征归一化、卷积、分组线性、五个 GRU、频谱滤波和 overlap-add 全在 GPU 执行，保留 960 点混合基 FFT、常驻权重与双状态缓冲。
- facade 增加 `gl`、`gles`、`vk`、`wgpu`、`dx` features 和对应模块的 `DFN3`；CLI 接入 `--backend ort|gl|gles|vk|wgpu|dx`，`--provider` 仅适用于 ORT。默认构建和 Release ZIP 仍使用 ORT。
- 四种后端在原生 Windows / RTX 2080 Ti 上通过独立 ONNX 连续帧数值基准、跨线程使用、强度切换、尾帧、无效输入、reset，以及真实 raw/WAV、独立立体声与批处理验收。WSL Ubuntu 24.04 通过 Compute-only 编译和 44 项非 GPU 测试，6 项硬件测试保留 ignored；Linux GPU 路径和 macOS 未验证。
- CI 分开运行普通工作区测试和 ORT 的真实模型测试，新增 Compute-only CLI 非 GPU 验收；不再在无 GPU runner 上强行运行标为 ignored 的硬件测试。

## 0.1.0

这是三层库与独立 CLI 的初始版本；此条目不表示已上传 crates.io。

- `anr-core` 提供独立于推理运行库的 `NoiseReduction: Send`、`Error` 和 `Result`。
- `anr-ort` 提供有状态的 DFN3 ONNX Runtime 后端，支持模型文件或自定义 session，构造时检查具名 `f32` 三入三出的维数及已知静态长度，兼容一维符号维度以及衰减量、`lsnr` 的标量形式。
- DFN3 接收 48 kHz 单声道采样，每帧 480 个采样、状态长度 45,304；尾帧补零并返回原始长度，`reset` 保留强度。
- DFN3 的强度范围为 0–100，映射到 0–60 dB 的最大衰减量；NaN 忽略，无穷和越界值钳制。
- `anr` 作为统一入口，默认不启用后端；提供 `ort`、`ort-load-dynamic` 及 CUDA、TensorRT、DirectML、CoreML、XNNPACK 编译选项。
- 固定使用 `ort 2.0.0-rc.15` 与 API 27，避免 rc.13 的失败加载全局状态污染；仅支持最新 Rust stable，不维护固定 MSRV，不启用 `ndarray`。默认预编译运行库为 1.31.0，自备 API-27-compatible 1.28.0 仍可使用。
- 提供原始小端序 `f32` 文件处理示例，以及独立 crate 的 README 和 MIT OR Apache-2.0 许可文本。
- 新增 `anr-cli`（二进制名 `anr`）：通过统一 `anr` 入口执行文件降噪、串行批处理、音频信息查询和 raw f32le stdin/stdout；默认启用 CPU ORT，provider 显式选择且不静默回退。
- CLI 采用 Symphonia 解码、rubato 带限流式重采样、每声道独立模型状态、跨解码包模型分帧，以及 PCM16/PCM24/float32 WAV 或 raw 输出；默认保留采样率、声道和帧数，不猜测性补偿模型算法延迟。
- CLI 默认不覆盖目标，完成并 finalize 后才提交同目录临时文件；批次单文件失败继续、成功结果保留。音频 stdout 与 stderr 诊断分离，支持取消与下游 BrokenPipe。
- CI 配置覆盖 Windows x64、Linux x64、macOS arm64/x64 四种原生 runner，统一使用 Rust stable；包含固定模型下载校验、真实模型测试、无后端模式、Clippy、全 features 编译、文档和解压 ZIP 后的实际推理。配置不等于平台通过，结果仍以实际 job 为准。
- CI 在 main/master push、版本 tag、pull request 或手动操作时运行；只取消同一 PR 的旧任务，普通 push 使用每提交独立的运行，不静默丢弃较早提交的预发布。
- CI 使用按平台、rustc 版本和工作流配置区分的 Cargo 缓存，包含 ORT 下载目录；Intel macOS 的 CPU 共享库跨运行复用，缓存命中时跳过 ORT 源码构建，仍执行所有 Rust 构建与测试。
- 强度上限测试使用 `1e-5` 数值容差，兼容 ARM64 CPU 归约的浮点舍入差异；不改变实际降噪计算。
- 补充 CLI 独立许可和第三方说明，明确 Symphonia MPL-2.0 源码获取/修改义务，及 Rust ORT 绑定、原生运行库与模型权重的许可边界。
- 发布安全修复：按实际 FileId 保护输入/模型和本批已完成结果，不再依赖操作系统名称或字符串大小写判断；写入前、提交前均复核，已保存但身份记录失败时明确报告状态并停止后续处理。
- WAV EOF 校验使用固定宽度 codec 的可靠声明帧数与实际解码帧数，拒绝无 payload 或尾部截断；未知长度标记不强造长度预算。
- 动态库使用可失败的显式加载入口；库构造改为 `DFN3::from_file(&Environment, path)`，应用持有环境，CLI 整批共用一个环境；`NoiseReduction` trait 与 CLI 命令不变。
- CPU 构造显式选择内置 provider，不继承自动 NPU 策略，也不强制 minimal runtime 实现设备发现；JSON 输出保留 BrokenPipe 分类。
- 拆分 `ort` 与 `ort-download-binaries`：普通默认用法仍下载 CPU 库，默认 facade 仍只依赖 core，关闭 defaults 的动态模式不再带下载器和 TLS/OpenSSL 构建依赖。
- GitHub 模型资产 Release 单独提供固定 SHA-256 的 `dfn3.onnx`、两份模型许可、NOTICE、清单及 SHA256SUMS；维护者已确认这份模型可按 MIT OR Apache-2.0 分发。
- main/master push 在四平台全通过后生成不可变提交预发布；`v<工作区版本>` tag 发布对应版本，拒绝版本错配；PR 和手动 CI 不发布，不涉及 crates.io。
- GitHub 应用 ZIP 包含 CLI、CPU ONNX Runtime 1.28.0、DFN3 模型及项目/模型/原生库许可；使用 `ort-load-dynamic`，不要求用户自行配置运行库搜索路径，`--model` 仍明确必填。
- 发布先上传到 draft，全部 ZIP 和 SHA256SUMS 核对完成后才公开；不移动 tag，不覆盖已公开版本或同名异摘要资产。使用内置 `GITHUB_TOKEN`，仅发布 job 授予 Contents 写入，无需专用 Secret；历史 workflow 提交若触发额外授权限制，明确失败而不替换目标提交。
- 发布前通过精确 Git ref 判断 tag 是否存在，避免首次预发布被不存在 ref 的 `422` 响应阻断；已有 lightweight/annotated tag 按完整 `refs/tags/…` 解析并校验提交。

完整模型不打包进 Rust 源码包；当前默认构建通过上述 features 自动准备模型，工作区仍可显式使用 `models/dfn3.onnx`。固定 GitHub 模型资产与应用 ZIP 的分发授权不代表任意来源模型的音质、性能或许可保证。
