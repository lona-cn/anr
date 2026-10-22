# anr-cli 架构与设计

`anr-cli` 是文件和标准流降噪前端，包名为 `anr-cli`，二进制名为 `anr`。本文描述当前实现的边界和数据流；命令、安装及部署参考见 [README.md](README.md)。CLI 不管理实时音频设备、虚拟麦克风或驱动，也没有动态插件 ABI。

## 模块边界

CLI 只通过 `anr` facade 创建降噪器，不直接依赖各执行后端。解码、重采样、声道路由、批次与文件事务属于 CLI；`anr-core::NoiseReduction` 只承担采样处理、强度设置和状态重置。

| 模块 | 职责 |
| --- | --- |
| [`main.rs`](src/main.rs) | clap 解析、取消处理器、错误分类与进程退出码；通过正常退栈释放当前输出。 |
| [`args.rs`](src/args.rs) | 子命令、值域、默认值，以及 stdin/stdout、格式和后端专用参数的组合校验。 |
| [`commands.rs`](src/commands.rs) | `denoise` / `inspect` / `backends`；模型选择、作业规划、目标保护、串行批次与最终提交。 |
| [`engine.rs`](src/engine.rs) | 静态 feature 对应的后端工厂、ORT provider 配置、`EngineSpec`、每声道 `EnginePool`。 |
| [`cancel.rs`](src/cancel.rs) | 共享取消标志检查和可识别的 `Cancelled` 错误。 |
| [`audio/mod.rs`](src/audio/mod.rs) | `AudioSpec`、`AudioInfo`、`AudioSource`、`AudioSink` 和 `ProcessStats`。 |
| [`audio/input.rs`](src/audio/input.rs) | Symphonia 探测/解码、音轨选择、raw f32le 读取、输入完整性与规格检查。 |
| [`audio/resample.rs`](src/audio/resample.rs) | rubato 分块重采样、延迟补偿、尾部排出和整数帧数换算。 |
| [`audio/pipeline.rs`](src/audio/pipeline.rs) | 声道独立模型帧、跨包累积、流尾补零和最终输出长度约束。 |
| [`audio/output.rs`](src/audio/output.rs) | WAV/raw 编码、同目录临时文件、提交和 raw stdout。 |
| [`audio/stdio.rs`](src/audio/stdio.rs) | 标准流的有界工作线程适配，使主处理栈可以响应取消。 |

路径参数使用 `PathBuf` / `OsString`，不要求普通音频文件名为 UTF-8。模型准备与部分后端仍有各自约束：构建时记录的模型源路径需为 UTF-8，ORT GPU 内存加载也要求模型目录可转换为 UTF-8，以解析外部权重目录。

## 命令与初始化顺序

三个子命令共享进程入口，但初始化范围不同：

- `inspect` 打开带头音频并读取所选音轨信息。规格缺失时可能需要先解码一个包；它不遍历整段录音、不验证整文件完整性，也不加载模型。时长/帧数来自容器声明，未知值在 JSON 中为 `null`。
- `backends` 返回全部十个后端的编译状态。ORT 项再列六种 provider；所有 `runtime_available` 均为 `null`，不初始化运行库或设备。
- `denoise` 先完成参数、目标和全局模型初始化，再创建输出并处理文件。

`denoise` 的实际顺序为：

1. `args::DenoiseArgs::validate` 拒绝不合法的参数组合，例如多输入共用 `--output`、raw 缺少规格、向终端发送二进制、非 ORT 后端显式传入 `--provider`。
2. 选择模型：显式 `--model`，否则由 `anr::default_model_path` 先找可执行文件旁的 `models/dfn3.onnx`，再找构建时记录的模型源文件。选择已有模型后，加载失败不换另一模型。
3. `plan_jobs` 生成输出名并预检目标，保护本次所有输入和所选模型。`--model -` 是普通文件名，只有音频输入/输出的 `-` 才是标准流标记。
4. `EnginePool::new` 初始化所选后端资源和一个模型实例。ORT 在此加载运行库并创建整个批次共用的环境。此阶段失败时尚未创建输出目录或临时文件。
5. 使用 `--output-dir` 时创建目录，并重新检查目标，防止原先不存在的父目录解析出新的路径别名。
6. 每个作业先复核目标，再打开输入、获得声道规格并调用 `EnginePool::prepare`。所需的全部声道实例就绪后才创建 sink。
7. 运行管线，检查取消标志，复核目标，调用 `sink.finish()` 提交，然后记录完成文件的身份。

没有全批次预解码。输入损坏可在前面文件已成功保存后才被发现；这不会撤销已完成结果。

### 模型准备与后端选择

CLI 默认 features 是 `ort-download-binaries`、`download-models`、`copy-models`。模型由 `anr/build.rs` 在构建期准备：下载固定资产并校验摘要，或使用 `ANR_MODEL_PATH`；可按 feature 复制到构建目录。运行期只查找本地文件，不下载。模型不是嵌入可执行文件的字节数组。应用 ZIP 的模型和运行库由打包脚本收集。

后端选择是编译期 feature 加运行期参数，不是插件加载。`--backend` 默认始终为 `ort`；即使只编入某个原生后端，也必须显式选择它。无 feature 构建保留 `inspect` 和 `backends`，但不能降噪。

`BackendResources` 封装十种后端的资源和构造差异，`EngineSpec` 提供管线使用的模型采样率、帧长。当前各 DFN3 后端均为 48,000 Hz、480 采样、单声道状态；解码和输出模块不依赖后端常量。

ORT provider 与原生后端是两条独立路径：

- ORT 默认为 CPU，其他 provider 需要对应 `ort-*` feature、平台和运行库。provider 显式注册失败或 session 初始化失败会返回错误；允许 ORT 将不支持或按分配规则更适合 CPU 的算子放在 CPU 上。
- ORT CUDA、TensorRT、DirectML 通过 `anr-ort::session_from_file` 在内存中准备模型，原始模型不变。CPU、CoreML、XNNPACK 保留文件加载路径。图变换由 `anr-ort` 维护，CLI 不复制该逻辑。
- 原生 `gl`、`gles`、`vk`、`wgpu`、`dx`、`cuda`、`tensorrt`、`openvino`、`opencl` 不启用 ORT，也不接受 `--provider`。它们要求固定原模型摘要；其中 OpenVINO 显式默认 CPU，`--openvino-device` 可选择 GPU/GPU.n/NPU，不使用 AUTO/HETERO。

TensorRT/OpenVINO 的私有 shared shim 是后端内部的 SDK 适配，并不是公共插件 ABI。构建时 SDK、运行库/驱动、ORT provider 和模型授权是独立部署问题，详见 [README.md](README.md) 与 [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)。

## 流式音频管线

```mermaid
flowchart LR
    Source[文件或 stdin] --> Decode[Symphonia 或 raw f32le]
    Decode --> Planar[可复用 planar 缓冲]
    Planar --> ToModel[需要时重采样到 48 kHz]
    ToModel --> Frames[跨包积攒 480 采样]
    Frames --> Reduce[每声道独立模型状态]
    Reduce --> ToOutput[需要时重采样到输出采样率]
    ToOutput --> Limit[约束到实际输入时长]
    Limit --> Encode[WAV 或 raw f32le]
    Encode --> Sink[临时文件提交或 stdout]
```

`AudioSource::read_frames` 填充可复用的每声道 `Vec<f32>`，返回本次每声道有效采样数，0 表示 EOF。`AudioSink::write_frames` 接收同样的 planar 布局；raw 读取/写入适配器负责与交错小端 `f32` 转换。这里的音频帧指同一时刻每声道各一个采样，不是压缩包或模型帧。

解码、模型和编码的块长互不绑定。输入读取以每次最多 4096 帧向管线交付数据，内部允许缓存一个解码包；模型按 480 采样消费，输出按有限块序列化。音频缓冲规模由声道数、采样率比、解码包和固定工作块决定，不随录音时长累积。模型实例和作业列表另随声道数、输入文件数增长。

### 输入规格与完整性

Symphonia 使用启用的 `all` codec/container registry。容器名来自实际探测标记，而不是扩展名；选择第一个能创建解码器的音轨。其他音轨的包被跳过。选中后不自动切换音轨；demuxer 请求 reset 或音频规格发生变化均为错误。

raw 输入明确提供采样率和声道数，支持短 read 和跨 read 的半个采样。只在 EOF 仍留有不完整音频帧时判为截断。所有输入、输出采样都必须有限。

固定宽度 WAV 根据可靠的声明帧数维护剩余预算，提前 EOF 报错；未知长度标记及其存在歧义的边界值不作为精确预算。其他容器的时长估算不用于强制补齐或截短音频。该检查不能识别所有容器/codec 损坏。

编码 stdin 经不可 seek 的 `MediaSource` 交给 demuxer。需要 seek 的容器可能失败，不以整流内存缓存或临时磁盘副本掩盖这一限制。流式容器读入的新标签只保留最新 metadata revision，避免历史封面和标签随流增长。

### 重采样、模型帧和时长

`StreamResampler` 在采样率相同时直接转发，否则使用 rubato `FftFixedInOut<f32>`。FFT 计划、输入/输出缓冲和模型帧缓冲在每条管线开始时构造，处理期间复用。通过 `process_into_buffer` 处理整块并丢弃 `output_delay()` 指定的滤波延迟。EOF 使用可复用补零块排出滤波尾部，再按目标帧数裁剪，不调用会分配临时 planar 输入的 partial 接口。

`ModelFrames` 跨解码包和重采样块保留未满帧采样，只有整个流结束才补零运行最后一帧。模型获得完整 480 采样，但只有最后一帧的有效部分继续送入输出阶段。不会在每个解码包末尾补零，也不会每包 reset。

总帧数由实际解码结果累计，使用 `u64` 计数、`u128` 乘法换算：

- 中间模型域目标为 `ceil(N_in × model_rate / input_rate)`，保留最后不足一个模型采样的时长，避免两次取整丢失最终输出采样。
- 最终目标为 `round(N_in × output_rate / input_rate)`，半入；保留原采样率时恰好为 `N_in`。
- `LimitedWriter` 随实际已读入的输入时长更新写出上限，只保留中间 ceil 引入的有限超前采样；EOF 再约束到最终目标并检查实际写出数相等。该机制也约束不可回退的 stdout。

重采样器滤波延迟和模型算法延迟不是一回事。`NoiseReduction` 没有模型延迟描述或 end-of-stream flush 接口，CLI 不猜测 DFN3 算法延迟，不移位补偿，也不额外执行模型尾部来声称恢复所有尾音。等长输出只说明采样计数一致，不说明模型无延迟。

### 声道与文件状态

`EnginePool` 为每声道持有独立 `Box<dyn NoiseReduction>`，不将左右声道轮流送入同一实例。声道增加时加载额外实例，减少时截掉多余实例；后续文件可以复用保留下来的实例。创建额外实例失败会终止批次。

`pipeline::process` 在每条流开始时对所有引擎执行一次 `reset()`。同一文件内连续调用保留状态，文件之间不继承循环状态。每个文件重新构造两级重采样器、模型帧累积器和输出计数器。声道数量与排列保持不变，不隐式混音。

## 输出事务与目标保护

### 文件输出

`FileSink` 在最终目标所在目录创建 `NamedTempFile`，以 `BufWriter` 写 raw 或由 hound 写 WAV。WAV 编码支持 PCM16、PCM24、float32，整数采样舍入并限制到可表示范围；每次写入检查 RIFF 32 位长度上限，不生成 RF64。任一次写入/结束失败会锁住 sink，禁止随后提交失败文件。

`finish()` 先 flush raw 或 finalize WAV，再检查取消并提交：

- 默认 `persist_noclobber`，即使目标在处理期间被其他进程创建，也不以覆盖方式提交。
- `--overwrite` 使用 `persist` 替换目标，成功前旧文件保持原状。同目录临时文件避免跨文件系统搬运；替换依赖文件系统提供的原子语义。
- 出错时 `PersistError` 仍持有临时文件，退栈释放后清理。管线本身不调用 `finish()`，只有命令层确认全流成功才提交。

这是单文件提交策略，不是整批事务，也没有 `fsync`/目录同步带来的断电持久性保证。`--output-dir` 创建父目录；单文件 `--output` 不替调用者创建父目录。

`DestinationGuard` 同时保存解析路径和 `file-id` 文件身份，保护所有输入、选中模型和本批次已完成结果。已有路径尽量 canonicalize，不存在的目标尽量使用已解析父目录；因此目录创建后还要重检。处理前和提交前重复检查，覆盖大小写不敏感别名、符号链接与硬链接等常见冲突。默认 no-clobber 提交还负责处理目标创建竞争；这些用户态检查并不是抵抗任意恶意目录替换的文件系统沙箱。

每次提交后记录文件身份。若提交已成功但身份读取失败，CLI 明确报告“已保存”并停止，不能把该错误误报为输出未提交。

### stdout 与取消

stdout 仅支持 raw f32le，拒绝交互终端。WAV 需要 seek 回填文件头，CLI 不伪造长度，也不缓存整条音频以支持 WAV stdout。日志、进度、统计、错误全部写 stderr。shell 重定向创建的目标文件不属于 CLI 的事务保护。

`audio/stdio.rs` 使用命名工作线程、容量为 1 的请求/回复通道以及复用缓冲，在标准流阻塞期间让主栈每 25 ms 检查取消。stdin 适配器使用 64 KiB 缓冲；stdout 直接复制 OS handle，避免全局 `Stdout` 行缓冲在退出时再次阻塞。

取消时某个工作线程的 OS read/write 仍可能阻塞，但它只持有标准流，不持有文件事务。主处理栈不等它完成，就能退栈清理临时文件并返回。推理和普通文件 I/O 仍采用操作之间的协作取消，不保证中断后端正在执行的调用。

`main` 正常返回 `ExitCode`，不在持有临时文件的调用栈里直接 `process::exit`：

| 情况 | 行为 |
| --- | --- |
| 参数错误 | 退出 2。 |
| 全局初始化/新增声道初始化失败 | 停止批次，退出 1。 |
| 单文件解码、推理、编码或提交失败 | 清理当前临时文件，继续其余作业；最终退出 1。 |
| Ctrl+C / 取消标志 | 停止批次、清理当前临时文件，退出 130。 |
| 下游关闭 stdout 引发 `BrokenPipe` | 停止处理，退出 0；同样适用于 inspect/backends 的结构化输出。 |
| 全部完成 | 退出 0。 |

stdout 发出的字节不能撤回；即使 BrokenPipe 按成功退出，也不代表整段音频已完成。调用程序需根据退出状态及消费协议判断是否接受输出。

## 设计取舍与维护依据

- **原生 Rust 音频管线**：Symphonia、rubato、hound 分担解码、带限重采样和 WAV 编码，不把 FFmpeg/Python 变成用户运行依赖。代价是必须明确处理 packet/frame 边界和流尾计数。
- **静态后端扩展**：复用 `anr` 的 Cargo features 和统一采样接口，避免再引入一套插件 ABI；后端 SDK 的私有适配留在各后端 crate。
- **有界流式处理**：文件与管道共用数据流，不因录音时长线性增加音频内存。
- **单文件事务与纯字节 stdout**：文件允许延迟提交，标准流无法回滚，两者保留各自真实的错误语义。

[`tests/cli.rs`](tests/cli.rs) 以实际子进程覆盖参数、JSON、分片 raw、声道隔离、强度、重采样、尾帧、批次 reset、失败清理、别名保护、BrokenPipe 和动态库缺失等行为。`tests/fixtures/` 的小 ONNX 图是状态和 I/O 控制图，不是 DFN3 权重；真实原生后端测试通过 `ANR_DFN3_MODEL` 和 ignored 测试显式启用，需要对应硬件/SDK。这些测试的定义不是本次运行证据，也不替代感知音质评估。

[CI workflow](../../.github/workflows/ci.yml) 配置四个原生平台的 SDK-free 检查与 CPU ZIP smoke，TensorRT/OpenVINO SDK 构建及硬件测试另需配置环境。平台矩阵、运行库固定版本、打包和发布规则见 [RELEASING.md](../../RELEASING.md)；第三方许可、模型授权见 [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)。本文不保存机器验收日志或临时测试计数。
