# anr-cli 设计与实施计划

> 状态：CLI 已实现并完成 Windows 与 WSL Ubuntu 的本地验收。macOS 和 GitHub Actions jobs 尚未实际运行，不标记为通过；实际用法见 README，验收证据见文末。
> 已确认范围：文件降噪、批处理、stdin/stdout 管道。实时设备、虚拟麦克风和驱动不在本次范围。
> For agentic workers: execute the implementation tasks with `subagent-driven-development` or `executing-plans`. Do not create worktrees, commit, push, or publish without explicit user authorization.

**Goal:** 在 `crates/anr-cli/` 提供可脚本化、可扩展后端、支持 Windows / Linux / macOS 的文件与管道降噪工具。

**Architecture:** CLI 通过 `anr` 统一访问降噪器。音频解码、重采样、声道管理、文件输出和管道协议留在 CLI，不塞进 `anr-core` 或 ORT 后端。先采用静态编译的后端选择和内部模块边界，不引入动态插件 ABI。

**Tech Stack:** Rust stable / edition 2024；clap、Symphonia、rubato、hound、tempfile；现有 anr / DFN3 / ORT。

**Spec:** 本文件“范围与约束”“命令契约”“音频处理契约”“平台与分发”是实施任务的规格依据。

## 范围与约束

- 包名 `anr-cli`，二进制名 `anr`；命令行参数通过 `PathBuf` / `OsString` 处理路径，不要求路径可转换为 UTF-8。
- 保持项目代码 `MIT OR Apache-2.0`，模型文件不嵌入、不打包、不自动下载。
- 首个实现通过 `anr` 使用现有 ORT DFN3；不绕开统一入口直接依赖 `anr-ort`。
- CPU 是三平台共同的默认执行方式；GPU provider 是显式选择、按 feature 构建的能力，不自动切换或静默退回 CPU。
- 文件、管道共用同一条有界缓冲处理管线；内存随声道数、解码包和固定处理缓冲区增长，不随录音时长累积。
- 不将 FFmpeg 或 Python 作为安装和运行的必需项；可以通过 raw PCM 管道与它们配合。
- 不承诺实时音频、虚拟声卡、云端服务、动态插件、自动模型搜索和模型下载。
- 不声称计划中的格式、命令或平台已经通过验证；实际交付时逐项记录运行证据。

## 现有接口事实

- `anr-cli/` 已加入 workspace，包含完整命令、音频管线、行为测试和独立发布文件。
- 根 workspace 包含 `anr-core`、`anr-ort`、`anr`、`anr-cli`，公共版本为 `0.1.0`。
- `anr::NoiseReduction: Send` 提供 `run(sample_rate, input, output)`、`factor`、`set_factor`、`reset`；接口不负责文件、重采样或多声道路由。
- 当前 DFN3 要求 48,000 Hz、单声道 `f32`，每帧 480 个采样，跨调用保持状态；每次不足一帧时都会补零并推进完整一帧。
- 当前 `crates/anr/examples/denoise.rs` 是整文件读取的 raw f32le 示例，不是本 CLI 的实现，也不是后续流式实现的内存模型。
- `models/dfn3.onnx` 可用于本地真实验收；其来源和使用授权与项目代码许可分别处理。

## 命令契约

采用子命令，而不是把所有未来功能堆在顶层参数中：

```text
anr denoise INPUT... --output OUTPUT --model MODEL [OPTIONS]
anr denoise INPUT... --output-dir DIRECTORY --model MODEL [OPTIONS]
anr inspect INPUT [--json]
anr backends [--json]
anr --help
anr --version
```

### denoise

建议参数：

| 参数 | 契约 |
| --- | --- |
| `INPUT...` | 一个或多个输入路径；`-` 代表 stdin。stdin 只能用于单输入任务。 |
| `--output PATH` | 单输入输出文件；`-` 代表 stdout。与 `--output-dir` 互斥。 |
| `--output-dir DIR` | 批处理输出目录；默认生成 `stem.wav`，显式选择 f32le 时生成 `stem.f32le`，同名目标不自动覆盖或任意重命名。 |
| `--model PATH` | 必须明确指定模型文件；不能依赖源码仓库当前目录下恰好有 `models/`。 |
| `--backend ort` | 当前只列出实际编译的后端；未来后端接入集中在 engine 模块。 |
| `--provider cpu` | 默认 CPU；CUDA、TensorRT、DirectML、CoreML、XNNPACK 按编译 feature 和平台显式开放。 |
| `--strength 0..100` | DFN3 的强度；拒绝 NaN、无穷和越界命令行参数，不能悄悄修正用户输入。 |
| `--input-format auto\|f32le` | 默认探测带头音频；raw PCM 必须显式选 `f32le`。 |
| `--input-rate HZ` | raw PCM 必填；不能用它覆盖带头文件的真实采样率。 |
| `--input-channels N` | raw PCM 必填；数据按声道交错排列，完整音频帧必须包含 N 个采样。 |
| `--output-format wav\|f32le` | 文件可由扩展名识别，也可显式指定；未知扩展名必须显式指定。stdout 必须明确选 `f32le`。 |
| `--output-rate HZ` | 默认保留输入采样率；显式指定时输出该采样率。 |
| `--wav-format pcm16\|pcm24\|float32` | WAV 编码，默认 PCM16；不能在 raw 输出中被静默忽略。 |
| `--overwrite` | 明确允许替换目标文件；仍禁止覆盖本次任意输入文件。 |
| `--quiet` / `--verbose` | 控制 stderr 的进度与诊断，不改变 stdout 音频数据。 |

拟定使用方式：

```text
anr denoise voice.wav --output clean.wav --model models/dfn3.onnx
anr denoise a.flac b.mp3 --output-dir cleaned --model models/dfn3.onnx
anr denoise - --input-format f32le --input-rate 48000 --input-channels 1 --output - --output-format f32le --model models/dfn3.onnx
```

第三条只表示程序参数。实际 stdin/stdout 由调用程序或二进制安全的 shell 管道连接，不把终端文本当 PCM。

批处理规则：

- 所有参数、模型加载和显式同名输出冲突先检查，再开始写输出。
- 默认不递归扫描目录、不在程序内部解释 glob；接收明确输入路径列表，避免三平台 glob 行为不一致。
- 逐文件串行处理，复用相同配置的降噪实例并在新文件开始前 reset；声道布局改变时调整实例集合。
- 单个输入解码失败：记录文件及原因，清理该文件临时输出，继续其他文件，最终非零退出。
- 参数、模型或 provider 初始化失败属于全局错误，不能继续生成貌似成功的结果。
- 成功文件保留；不承诺整批任务的事务回滚。提交文件时仍必须依靠不覆盖的文件系统操作处理竞争和平台特有的重名情况。

### inspect

读取音频容器和音轨信息，不加载降噪模型。输出格式、所选音轨、采样率、声道数；时长未知时明确为未知，不能输出假精确值。`--json` 使用稳定字段，未知字段为 `null`，诊断仍写 stderr。

多音轨输入默认选择第一个可解码的音频轨，并明确显示选择；不把音轨切换或中途采样率/声道变化悄悄当成同一条连续音频流处理。

### backends

列出当前二进制编译的后端与 provider，并区分“已编译”和“运行时可用性未检查”。只有实际调用运行库查询后才能报告 provider 的可用性；它也不等于模型已在设备运行成功。

### stdout、错误和取消

- `denoise --output -` 的 stdout 只能包含完整的 raw f32le 音频字节。日志、进度、错误、统计全部进入 stderr。
- 不默认向交互终端输出二进制音频；要求重定向或改用文件。
- WAV 文件头需要 seek 回填，本次不向不可 seek 的 stdout 输出伪造长度的 WAV，也不为此缓存完整音频。
- 退出码：成功、help/version 为 0；运行/部分批处理失败为 1；参数错误为 2；主动 Ctrl+C 取消为 130。
- 下游正常提前关闭管道造成的 `BrokenPipe` 按消费完成处理，不输出 panic 或冗余错误栈。
- 用跨平台中断处理设置取消标志，在处理循环退出并释放资源后返回退出码；不能在持有临时文件的调用栈中直接 `process::exit`。
- 文件输出能在完成前保持原目标不变；stdout 已发出的音频无法撤回，出错时必须以退出码和 stderr 表示不完整流。

## 音频处理契约

```mermaid
flowchart LR
    Source[文件或 stdin] --> Decode[解码或 raw PCM 读取]
    Decode --> Layout[按声道整理采样]
    Layout --> ToModel[需要时重采样到模型采样率]
    ToModel --> Frames[跨解码包累积完整模型帧]
    Frames --> Reduce[每声道独立降噪状态]
    Reduce --> ToOutput[需要时重采样到输出采样率]
    ToOutput --> Encode[WAV 或 raw f32le 编码]
    Encode --> Sink[文件提交或 stdout]
```

### 输入与输出格式

- 带头文件使用 Symphonia 0.5.5 的容器探测和 codec registry；启用已有解码能力，不自行实现压缩音频解码器。
- 拟支持 WAV/PCM、FLAC、MP3、OGG/Vorbis、AAC-LC/M4A。对 AAC profiles、Opus 等未被所选 codec 支持的组合明确报错，不能把“可识别容器”误称为“能解码所有音轨”。
- 非 seek 的编码 stdin 受具体 demuxer 限制；raw f32le 是保证稳定的管道协议。需要 seek 的容器应返回具体原因，不把整条 stdin 读入内存或暗中落盘。
- raw f32le 按小端 IEEE 754 解析，支持一次 read 只返回半个采样的情况；只有 EOF 时仍有残余字节或不完整声道帧才算截断。
- 文件输出提供 PCM16、PCM24 和 float32 WAV，以及 raw f32le；本次不增加 MP3/AAC 编码器。
- 默认保留声道数和排列，不暗中混成单声道。每个声道创建独立的 `NoiseReduction` 状态；一个实例不能轮流处理左、右声道。
- WAV 首次实现只写音频和必要头部，不声称保留原封面的标签、章节、容器元数据或特殊声道布局标记。

### 分帧、重采样和时长

- 引擎工厂提供内部 `EngineSpec { sample_rate, frame_samples }` 和降噪实例。管线使用该规格，不在解码、输出模块散布 DFN3 常量。
- 模型初始化、FFT/重采样器构造和缓冲区分配发生在处理循环外；输入、输出和各声道缓冲循环复用。
- 对 48 kHz 输入/输出绕过不需要的重采样；其他采样率使用 rubato 的带限流式重采样器，不手写线性插值取代抗混叠滤波。
- 解码包和模型帧不是同一个概念：跨包积累完整的 480 采样模型帧，只有整个音频流结束才补最后一帧。
- 使用 rubato 的 `process_into_buffer`、`output_delay` 和尾部处理接口。补偿重采样器滤波延迟，耗尽它的尾部，再按整数采样计数裁剪，不能靠目测丢掉开头或结尾。
- 默认保持原采样率时，输出每声道采样数必须等于实际解码的输入每声道采样数；不能依赖容器声明的近似时长。
- 显式输出其他采样率时，输出帧数为 `round(N_in * rate_out / rate_in)`，使用足够宽的整数累计计算，不能逐包取整累积漂移。
- 连续调用不 reset；新文件、新流或重新建立管线时重置模型和重采样状态。
- 重采样器延迟与 DFN3 算法延迟是两回事。现有后端没有模型延迟描述或流尾 flush 契约，本 CLI 不擅自猜测延迟值或声称已补偿 DFN3 算法延迟；该限制必须进入使用文档。

### 文件写入

- 文件 sink 在目标目录创建临时文件，完成推理、编码和 WAV finalize 后才提交；默认使用 `persist_noclobber`。
- `--overwrite` 只影响最终提交，不允许在处理开始前截断旧文件，也不能覆盖本次输入路径。
- 模型、解码、写入、取消等错误都释放并清理当前临时文件；前面成功的批处理输出不删除。
- 临时文件与目标同目录，避免跨文件系统移动；不宣称提交同时保证断电持久性。

## 模块与依赖

拟定文件边界：

```text
crates/anr-cli/
  Cargo.toml
  README.md
  DESIGN.md
  LICENSE-MIT
  LICENSE-APACHE
  src/
    main.rs                 # 进程入口、取消、退出码
    args.rs                 # clap 命令与互斥/依赖参数
    commands.rs             # denoise/inspect/backends，批处理编排
    engine.rs               # anr 后端、provider、EngineSpec、每声道实例
    audio/
      mod.rs                # 内部音频规格和共享数据类型
      input.rs              # Symphonia 解码与 raw Read 适配
      output.rs             # WAV/raw Write 与文件提交
      resample.rs           # rubato 缓冲、延迟和尾部处理
      pipeline.rs           # 声道状态、模型帧累积、采样计数
  tests/
    cli.rs                  # 真实进程的音频/文件/管道契约
    fixtures/               # 自制微型 ONNX 控制图，不是 DFN3 权重
```

依赖基线及理由：

| 依赖 | 建议基线 | 用途与边界 |
| --- | --- | --- |
| `anr` | workspace 0.1.0 | 唯一的降噪库入口；CLI 默认启用 CPU ORT，并转发现有 provider features。 |
| `clap` | 4.6.7 | 参数解析和帮助；已核对其声明的 MSRV 为 1.85。 |
| `symphonia` | 0.5.5 | 纯 Rust 解码；使用明确的 codec/features 集合，MPL-2.0。 |
| `rubato` | 0.16.2 | 纯 Rust 带限分块重采样；已核对 MSRV 1.61。 |
| `hound` | 3.5.1 | WAV 编码及 finalize；不用于向 stdout 伪造可 seek 的 writer。 |
| `tempfile` | 3.27.0 | 同目录临时输出及不覆盖提交；已核对 MSRV 1.63。 |
| `serde` / `serde_json` | 1.x | inspect/backends 的结构化输出；不与音频 stdout 混用。 |
| `ctrlc` | 3.x | Windows / Unix 的主动取消通知，不在信号回调里执行推理或文件提交。 |

这里是已调查 API 的依赖基线，不是要求为所有普通依赖使用精确 `=` 锁定；实施时使用 Rust stable 解析兼容版本、更新 Cargo.lock 并构建验收。ORT 继续遵循 workspace 已有的精确预发布版本约束。

Symphonia 的 MPL-2.0 需要单独记录，不能因为项目采用 MIT OR Apache-2.0 就删除依赖自身的许可和来源说明。二进制分发要记录 codec、重采样器、ORT 和模型的独立许可要求，不把未知权重纳入项目授权。

## 平台与分发

目标发布矩阵，不等于本轮已验证：

| 系统 | 首批目标 | 验收关注点 |
| --- | --- | --- |
| Windows | `x86_64-pc-windows-msvc` | 原生二进制管道、Unicode/空格路径、输出占用及不覆盖语义、CPU 运行库。 |
| Linux | `x86_64-unknown-linux-gnu` | stdin/stdout、BrokenPipe、权限和符号链接、构建产物的 glibc 基线。 |
| macOS | `aarch64-apple-darwin`、`x86_64-apple-darwin` | 两种架构分别运行、动态库查找、CPU 后端；CoreML 为显式可选能力。 |

- 使用 std 的路径与二进制 I/O，不用 Bash 脚本、PowerShell cmdlet 或平台专有工具承担核心业务。
- Windows PowerShell 5.1 的文本管道不能当作透明二进制管道；文档使用 `cmd.exe` 或 PowerShell 7.4+ 的 native-to-native 字节管道示例。不要把 stderr 合并进音频 stdout。
- 编译某个 target 不等于支持该系统；Windows、Linux、macOS 都要原生启动二进制并处理音频。macOS 的两种架构也不能相互代替运行验收。
- Linux GNU 产物不宣称兼容 musl 或任意旧 glibc；最低 OS / 运行库要求随实际构建基线和运行验证记录。
- 发布压缩包包含 executable、所需运行库及适用许可说明，模型单独提供。通过 `cargo install anr-cli` 安装时也不能依赖工作区外路径。
- `.github/workflows/ci.yml` 仅使用 Rust stable 执行原生 OS 构建与测试；模型无再分发授权时 CI 使用自制控制图，本地额外运行用户提供的真实模型。
- 发布流程增加 `anr-cli`，顺序为 `anr-core → anr-ort → anr → anr-cli`；所有内部依赖同时带版本与本地路径。

CLI features 与 `anr` 保持同名转发，不维护第二套 provider 绑定：

```toml
[features]
default = ["ort"]
ort = ["anr/ort"]
ort-load-dynamic = ["ort", "anr/ort-load-dynamic"]
ort-cuda = ["ort", "anr/ort-cuda"]
ort-tensorrt = ["ort", "anr/ort-tensorrt"]
ort-directml = ["ort", "anr/ort-directml"]
ort-coreml = ["ort", "anr/ort-coreml"]
ort-xnnpack = ["ort", "anr/ort-xnnpack"]
```

## 实施顺序与验收

每个任务先写能捕获具体消费者可见错误的测试，再实现并运行对应命令。不为固定文案、源代码文本或纯转发编写测试。

### 任务 1：可安装命令与流协议

**文件**：根 `Cargo.toml`、`crates/anr-cli/Cargo.toml`、`src/main.rs`、`src/args.rs`、`src/commands.rs`、`src/engine.rs`、`tests/cli.rs`。

- [x] workspace 增加 CLI 成员和带 `version + path` 的 `anr` 依赖；package 继承版本、edition 和双许可，声明 `[[bin]] name = "anr"`。
- [x] 实现 denoise / inspect / backends 参数和上述互斥规则，定义统一退出码及 stderr 输出。
- [x] engine 工厂只使用 `anr`，对未编译或无法初始化的显式 provider 报错，不静默回退。
- [x] 用自制 ONNX 控制图运行真实 raw 输入/输出进程，断言音频值和完整字节数；stdout 不得混入日志。
- [x] 以 1、2、3 字节等短 read 粒度输入相同 PCM，证明跨字节边界还原正确；EOF 残余字节或声道帧必须失败。
- [x] 实际执行二进制的 --help、backends --json、48 kHz raw 文件和 stdin/stdout；安装后另从无关目录运行真实模型。

### 任务 2：完整文件音频管线

**文件**：`src/audio/{mod,input,output,resample,pipeline}.rs`，接入任务 1 的 denoise 命令，扩展 `tests/cli.rs`。

- [x] 实现 Symphonia 解码及实际音轨选择；inspect 与降噪共用探测结果，不能显示一种规格却处理另一条轨道。
- [x] 帧累积器跨解码包拼完整模型帧，仅 EOF 补尾帧。用 481 个采样、不同分包边界及多模型帧验证长度与输出一致性。
- [x] 每声道独立实例和状态；用左声道非零、右声道静音的输入及输入驱动的状态控制图，证明不发生跨声道状态污染。
- [x] 使用 rubato 复用缓冲，处理输入、输出采样率变化，显式清除滤波延迟并完成尾部排空。
- [x] 用 16 kHz、44.1 kHz、48 kHz 短音频验证默认采样率/声道/采样数保留，换输出采样率时满足整数帧计数规则；不同输入分包不能累积取整漂移。
- [x] 实现 WAV PCM16 / PCM24 / float32 写入和正确 finalize；对输出重新解码验证 header 与实际音频，不只断言文件存在。
- [x] 使用 `models/dfn3.onnx` 实际处理 WAV、非 48 kHz 音频、双声道和 raw 管道。验证有限输出、声道、时长和帧边界，不把非静音当作音质提升证明。

### 任务 3：批处理和安全文件输出

**文件**：`src/commands.rs`、`src/audio/output.rs`、`src/main.rs`、`tests/cli.rs`。

- [x] 预先检查参数、模型、输入/输出冲突；默认不覆盖，显式 overwrite 也禁止修改本次输入文件。创建输出目录后再次核验真实路径，模型 `-` 仍作为文件保护。
- [x] 完成同目录临时文件、WAV finalize 后提交及失败清理；验证已成功写入一帧后发生真正 ORT 推理错误时，旧目标仍保持不变。
- [x] 批处理复用模型但 reset 状态；同一文件单独处理与批次中处理结果一致，不能继承上一段录音。
- [x] 验证一个文件失败后其他文件继续、错误文件没有成功产物、成功文件保留且最终退出码为 1。
- [x] 以原生进程测试 stdin EOF、stdout 提前关闭和取消；Windows 私有控制台 CTRL_BREAK、Linux SIGINT 均验证退出 130、旧目标保留和临时文件清理。

### 任务 4：三平台交付与发布

**文件**：`crates/anr-cli/README.md`、两份 LICENSE、必要的第三方许可说明、根 README / CHANGELOG / RELEASING、CI workflow。

- [x] README 写清 raw 格式、批处理、provider、模型路径、WAV stdout 限制、模型算法延迟和平台验证边界。
- [ ] 四个平台的 GitHub 原生 runner 全部执行通过：CI 已配置；本机 Windows / WSL Ubuntu 已实跑，macOS 及 GitHub jobs 需对应环境实际执行，不以本地结果替代。
- [x] 历史验收曾在 Rust 1.88 完成完整工作区编译和测试；安装到临时目录后，从无关工作目录使用实际 DFN3 模型运行成功。该记录不代表当前支持要求。
- [x] fmt、默认及无后端 clippy、无后端测试、所有 features 编译检查、无同名冲突的库 Rustdoc 通过。
- [x] 四个 package 均经 Cargo 内建 staging registry 解包构建验证；CLI 包包含独立许可/通知，不包含用户权重和本机音频。
- [x] 四 crate workspace publish dry-run 全部通过，明确中止了上传；未实际发布。

上述任务按依赖顺序交付。任务 1 的 raw 路径只是任务 2、3 的集成入口，不能将它当作已完成文件/重采样/批处理需求的最终交付。

## 设计取舍

- **选原生 Rust 管线，不选“必须安装 FFmpeg 的壳”**：安装路径和错误处理更可控；换来的是需要认真实现重采样和流尾计数。FFmpeg 仍可通过标准 PCM 管道协作。
- **选静态后端扩展，不选动态插件**：复用已有 `anr` features，避免跨平台 ABI、加载路径和供应链复杂度。
- **选有界流式处理，不选整段读入**：文件和管道共用逻辑，避免长录音内存线性增长。
- **选稳定字节协议，不向 stdout 混排进度或不完整 WAV 头**：便于脚本和其他语言可靠集成。

## 调查来源

- 当前接口与限制：`crates/anr-core/src/lib.rs`、`crates/anr/Cargo.toml`、`crates/anr-ort/README.md`、`crates/anr/examples/denoise.rs`。
- [Symphonia 0.5.5：格式、codec、探测与 gapless 行为](https://docs.rs/symphonia/0.5.5/symphonia/)。许可与 MSRV 另已核对本地缓存 manifest。
- [rubato 0.16.2 Resampler：复用缓冲、output_delay 与尾部处理](https://docs.rs/rubato/0.16.2/rubato/trait.Resampler.html)。
- [tempfile NamedTempFile：persist_noclobber、清理及资源泄漏边界](https://docs.rs/tempfile/3.27.0/tempfile/struct.NamedTempFile.html)。
- [PowerShell 官方说明：7.4+ native 字节重定向/管道及 stderr 合并限制](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_redirection?view=powershell-7.5)。

## 本次验收记录

详细命令、环境和边界见根目录 [RELEASING.md](../../RELEASING.md) 的“本次本地验证”。Windows x64 / Rust 1.88 工作区 56 项通过；WSL Ubuntu 24.04 x64 / Rust 1.99 工作区 57 项通过，额外一项为 Unix SIGINT 进程测试，均为 0 项忽略。已完成真实 DFN3 文件、管道、独立安装及四包发布 dry-run；没有把控制图当作降噪音质依据。

macOS 没有可用的本地执行环境；四平台 GitHub Actions 尚未触发，因此该条跨平台 runner 验收保留未勾选。Intel macOS 还需要 CI 中固定源码构建或自备的兼容 ORT 运行库，不能默认依赖不存在的预编译分发。

