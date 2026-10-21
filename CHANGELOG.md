# 变更记录

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
- CI 配置覆盖四种原生 OS/架构，统一使用 Rust stable 构建和运行进程测试，并验收安装后从无关工作目录处理控制图音频；配置不等于平台已通过，真实模型、格式和可选 provider 的运行结果须单独记录。
- CI 在 push、pull request 或手动操作时运行，使用只读仓库权限，并取消同分支的旧任务；执行格式、Clippy 和文档构建检查。
- CI 使用按平台、rustc 版本和工作流配置区分的 Cargo 缓存，包含 ORT 下载目录；Intel macOS 的 CPU 共享库跨运行复用，缓存命中时跳过 ORT 源码构建，仍执行所有 Rust 构建与测试。
- 补充 CLI 独立许可和第三方说明，明确 Symphonia MPL-2.0 源码获取/修改义务，及 Rust ORT 绑定、原生运行库与模型权重的许可边界。
- 发布安全修复：按实际 FileId 保护输入/模型和本批已完成结果，不再依赖操作系统名称或字符串大小写判断；写入前、提交前均复核，已保存但身份记录失败时明确报告状态并停止后续处理。
- WAV EOF 校验使用固定宽度 codec 的可靠声明帧数与实际解码帧数，拒绝无 payload 或尾部截断；未知长度标记不强造长度预算。
- 动态库使用可失败的显式加载入口；库构造改为 `DFN3::from_file(&Environment, path)`，应用持有环境，CLI 整批共用一个环境；`NoiseReduction` trait 与 CLI 命令不变。
- CPU 构造显式选择内置 provider，不继承自动 NPU 策略，也不强制 minimal runtime 实现设备发现；JSON 输出保留 BrokenPipe 分类。
- 拆分 `ort` 与 `ort-download-binaries`：普通默认用法仍下载 CPU 库，默认 facade 仍只依赖 core，关闭 defaults 的动态模式不再带下载器和 TLS/OpenSSL 构建依赖。

模型由使用者提供，不打包或自动下载。工作区可使用 `models/dfn3.onnx`，但该文件不随 crate 分发；本项目不对任意来源的模型作音质、性能或授权保证。
