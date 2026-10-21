# 变更记录

## 未发布

- 新增 `anr-gl`：固定 DFN3 模型的原生 OpenGL 4.3 Compute 后端，不依赖 ONNX Runtime；原 ONNX 仅作为按 SHA-256 校验的参数容器。
- FFT、特征归一化、卷积、分组线性、五个 GRU、频谱滤波和 overlap-add 均在 GPU 执行；采用 960 点混合基 FFT、GPU 常驻权重和双状态缓冲、每实例独立的 context 工作线程。
- facade 增加可选 `gl` feature 和 `anr::gl::DFN3`；CLI 的 `--backend ort|gl` 实际接入 engine 工厂，`--provider` 仅适用于 ORT。默认构建和 Release ZIP 仍使用 ORT。
- 保留原有分帧、补零、强度、重置和声道独立处理；覆盖独立 ONNX 数值基准、跨线程使用、强度切换、尾帧、无效输入、真实 raw/WAV/批处理。原生 Windows / RTX 2080 Ti 已实跑；WSL Ubuntu 24.04 的 GL-only 编译、非 GPU 测试和 Clippy 通过，Linux EGL GPU 路径尚未实跑，macOS 明确不支持。

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

Rust 包仍不打包或自动下载模型，工作区仍可使用 `models/dfn3.onnx`。GitHub 模型资产与应用 ZIP 的模型分发是单独的明确授权范围，不代表任意来源模型的音质、性能或许可保证。
