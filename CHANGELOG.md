# 变更记录

## 未发布

- `anr` 和 CLI 默认启用 `download-models`、`copy-models`：构建时从固定 GitHub Release 下载 DFN3，校验 SHA-256 并持久缓存，独立控制向 profile、examples、tests 构建目录复制模型；可用构建环境变量 `ANR_MODEL_PATH` 选择本地文件，关闭 defaults 则不引入模型下载器依赖。
- 新增 `anr::default_model_path()`，先查找可执行文件旁的 `models/dfn3.onnx`，再使用构建时记录的源路径。CLI 的 `--model` 改为可选，显式路径优先；自动选择的模型同样受输出覆盖保护。raw 示例也支持省略模型路径，运行时不下载模型。
- 新增工作区工具 `anr-bench`，串行比较 `anr-gl`、`anr-vk`、`anr-dx` 的连续帧推理性能；共用固定种子的 48 kHz 单声道输入，初始化单独计时，支持预热、后端选择、降噪强度和 JSON。报告同步调用延迟、nearest-rank P50/P95/P99、FPS、RTF 及超过 10 ms 的帧数；任一后端失败仍继续测试其余后端，并返回非零退出码。默认构建不依赖 ORT，不改动原生推理实现，也不加入应用 Release ZIP。
- `anr-bench` 增加可选的 `ort-cuda`、`ort-tensorrt`、`ort-directml` 后端，以及 `ort-download-binaries` / `ort-load-dynamic` 运行库来源选项；共用一个 ORT Environment，缺失运行库或 provider 作为单后端错误返回，新增 opt-in 硬件回归。
- 统一项目 ORT 默认策略：允许不支持或按 ORT 分配规则更适合 CPU 的算子使用 CPU kernel，CLI 和 benchmark 不再强制全算子 GPU；库保留调用方 Session 设置，provider 注册或初始化失败仍报错。benchmark 用 `cpu_fallback_allowed` 标注执行策略，耗时包含 CPU 算子及设备传输，不冒充纯 GPU kernel 时间。
- Windows / RTX 2080 Ti、匹配的 ORT 1.31 运行库下，当前 DFN3 已通过原生 GL/VK/DX 和 DirectML 混合执行 benchmark；CUDA 不支持原始模型中的 `FusedConv(Sigmoid)`，TensorRT 缺少 `/SequenceAt_11_output_0` shape，均保留实际初始化错误，不生成性能数字。

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
