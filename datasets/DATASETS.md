# 语音降噪数据集与下载入口

本页记录 20 项用于语音降噪、语音增强和去混响研究的数据集或数据集家族。同一家族的版本合并计数，编号不是质量排名。

清单区分成对语音、干净语音、背景噪声和房间脉冲响应（RIR）。有些发布完整音频，有些提供素材与合成脚本；CHiME-4 和 WHAM! 完整语音混合还涉及 LDC／WSJ0 授权。表中的采样率指发布格式或明确标注的录音规格，不代表有效带宽。

下载入口以官方发布页、仓库和论文为准。音频、元数据、上游素材与代码可能采用不同许可；使用和再分发前应保留对应版本的许可、署名、划分清单与校验文件。

## 语音增强训练与评测：6 项

| # | 数据集与官方来源 | 内容与采样率 | 下载入口与方式 | 许可、参考信号与限制 |
| --- | --- | --- | --- | --- |
| 1 | [DNS Challenge：DNS4／DNS5](https://github.com/microsoft/DNS-Challenge) | 面向 48 kHz 全频带增强的多语言干净语音、背景噪声和 RIR 素材；需合成 noisy/clean 对 | DNS4 使用[官方下载脚本](https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-4.sh)；DNS5 分[耳机语音](https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-5-headset-training.sh)、[免提语音](https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-5-speakerphone-training.sh)和[噪声／RIR](https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-5-noise-ir.sh)下载 | 各组成语料沿用[原许可](https://github.com/microsoft/DNS-Challenge#dataset-licenses)，不是统一 MIT 音频许可。按具体子集核对来源、带宽和用途 |
| 2 | [EARS／EARS-WHAM](https://sp-uhh.github.io/ears_dataset/) | EARS：48 kHz、约 100 小时、107 位英语说话人；消声室录音，含耳语、大声说话、情绪及自由表达 | [EARS 官方仓库](https://github.com/facebookresearch/ears_dataset)提供分说话人归档；[基准仓库](https://github.com/sp-uhh/ears_benchmark)提供 EARS-WHAM 与 EARS-Reverb 生成脚本 | CC BY-NC 4.0。EARS 是干净语音；EARS-WHAM 混入 WHAM! 噪声，EARS-Reverb 使用 RIR。记录基准版本：v2 调整了噪声／RIR 划分与预处理，不能与 v1 混报 |
| 3 | [VoiceBank+DEMAND](https://datashare.ed.ac.uk/handle/10283/2791) | 原版 48 kHz；28／56 位说话人训练版本；单通道增强基准 | 发布页分别提供 clean/noisy train、clean/noisy test ZIP；项目加载器使用 28-speaker 目录，见[当前项目的数据格式](#当前项目的数据格式) | 发布页附 CC BY 4.0 许可文本。语音来自 VCTK，部分噪声来自 DEMAND；为合成配对，不是现场同步采录的 dry/noisy 对。16 kHz 衍生版不符合本项目加载器要求 |
| 4 | [WHAM!／WHAMR!](http://wham.whisper.ai/) | 标准噪声包为 16 kHz、双声道、float32 WAV；另有原始 48 kHz 长录音噪声；WHAMR! 增加合成混响 | 官方站提供 WHAM! noise、WHAM!48kHz noise 归档，以及 WHAM!／WHAMR! 构建脚本和校验值 | 噪声为 CC BY-NC 4.0。生成完整语音混合需另行取得 WSJ0 授权；仅取得噪声不等于取得完整语音集。保留官方 train／validation／test 噪声划分 |
| 5 | [CHiME-4](https://www.chimechallenge.org/challenges/chime4/data) | 16 kHz；提供 1／2／6 通道任务；公交、咖啡馆、行人区和街道路口实录及模拟语音 | [下载说明](https://www.chimechallenge.org/challenges/chime4/download)将音频与 6 通道标注指向 CHiME-3 包 [LDC2017S24](https://catalog.ldc.upenn.edu/LDC2017S24)；1／2 通道补充标注另行发布 | 音频受 LDC 条款约束，不能套用 Apache 2.0 基线代码许可。模拟训练有 WSJ0 干净源；实录 CH0 是近讲麦克风，不是无噪、无混响的精确 dry reference |
| 6 | [Clarity：CEC2／ICASSP 2023](https://claritychallenge.org/docs/icassp2023/data/icassp2023_data) | 44.1 kHz、32-bit WAV；10,000 个模拟场景（6,000 train／2,500 dev／1,500 eval）；双耳多麦克风助听器增强，另有真实声学混合评测集 | [下载页](https://claritychallenge.org/docs/icassp2023/icassp2023_download)列出 core、train、HOA-IR 和 eval 包 | 组合数据，下载页未声明覆盖全包的统一许可；[OlHeaD-HRTF 上游](https://uol.de/mediphysik/downloads/hearingdevicehrtfs)为 CC BY-NC-SA 4.0，需核对各组件授权。train／dev 提供目标参考，不能假定公开 eval 也含同等参考信号 |

### 下载说明

- DNS4 脚本默认只读取 HTTP 响应头，DNS5 耳机脚本也默认 dry run；按脚本注释选择下载方式及所需分卷。DNS4 全量解压约需 1 TB 存储。
- DNS 仓库的 `noisyspeech_synthesizer.cfg` 默认 `sampling_rate: 16000`；准备 48 kHz 配对时必须显式配置采样率与输入路径，不能只依据 `fullband` 目录名。
- DeepFilterNet3 [论文第 4 节](https://arxiv.org/html/2305.08227#S4)使用 DNS4，并对 PTDB、VCTK 进行 10 倍过采样。其[DNS4 下载与 HDF5 转换脚本](https://github.com/Rikorose/DeepFilterNet/blob/main/scripts/download_process_dns4.sh)属于外部训练流程，不是 ANR-Lite-48 加载器的输入格式。

## 干净语音与设备退化：6 项

除 DAPS 同时提供设备退化版本外，本组主要作为人声参考来源，需要另行混噪或加入混响。

| # | 数据集与官方来源 | 内容与采样率 | 下载入口与方式 | 许可与使用限制 |
| --- | --- | --- | --- | --- |
| 7 | [VCTK v0.92](https://datashare.ed.ac.uk/handle/10283/3443) | 48 kHz、16-bit；110 位说话人；半消声室采录 | 爱丁堡 DataShare 提供 `VCTK-Corpus-0.92.zip`、README 和许可文本 | CC BY 4.0。部分说话人／麦克风录音有已知问题，按 README 筛选；应保留说话人级训练／测试隔离 |
| 8 | [PTDB-TUG](https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html) | 48 kHz、16-bit；20 位说话人、4,720 条录音；受控录音室采录 | 从官方页进入 [TU Graz 下载目录](http://www2.spsc.tugraz.at/databases/PTDB-TUG)；规格见[原始论文](https://www.isca-archive.org/interspeech_2011/pirker11_interspeech.pdf) | 官方明确数据库为 ODbL 1.0，单项内容为 Database Contents License（DbCL）1.0。降噪使用麦克风信号（`MIC`），不是喉电图（`LAR`） |
| 9 | [AISHELL-3](https://www.openslr.org/93/) | 44.1 kHz；约 85 小时、218 位普通话说话人；高保真近讲录音 | OpenSLR SLR93 提供 `data_aishell3.tgz` 及多个镜像 | Apache 2.0。适合补充中文人声；发布采样率不是 48 kHz，预处理时记录原始带宽。规格来源：[原始论文](https://arxiv.org/abs/2010.11567) |
| 10 | [Hi-Fi TTS](https://www.openslr.org/109/) | 44.1 kHz；约 291.6 小时、10 位说话人；经带宽和 SNR 筛选的 LibriVox 有声书 | OpenSLR SLR109 提供 `hi_fi_tts_v0.tar.gz` 及镜像；[论文](https://arxiv.org/abs/2104.01497)说明 `clean`／`other` 子集 | CC BY 4.0。降噪参考优先选 `clean`；`other` 的噪声和带宽条件较弱，不能把全库当同等级的原始录音室干净语音 |
| 11 | [LibriTTS-R](https://www.openslr.org/141/) | 24 kHz；约 585 小时、2,456 位说话人；经过语音恢复模型处理 | OpenSLR SLR141 按 train／dev／test 分卷下载，另有恢复失败样本清单 | CC BY 4.0。适合扩充说话人多样性和 24 kHz 训练；它不是未经处理的原始 clean recording，升采样不能补回高频 |
| 12 | [DAPS](https://ccrma.stanford.edu/~gautham/Site/daps.html) | 发布版 44.1 kHz、20 位说话人；每人 5 段文本，含 `clean raw`、`clean`、`produced` 及设备版本，见[原始论文](https://ccrma.stanford.edu/~gautham/Site/daps_files/mysore-spl2015.pdf) | 作者页指向 [Zenodo 记录](https://zenodo.org/records/4660670)；从该记录获取发布包与随附说明 | 作者页和论文未明确音频复用许可；使用前查随包条款或向作者取得授权，不以第三方镜像的许可标签为准。设备版为 `clean` 经扬声器回放后采录，含房间、设备及可能的设备预处理；`produced` 是制作后语音，不是 dry clean |

## 背景噪声与环境声音：6 项

本组不是 clean/noisy 语音对。环境音频可能夹有人声或音乐，需要按训练目标筛选，不能把所有类别都当作无语音背景噪声。

| # | 数据集与官方来源 | 内容与采样率 | 下载入口与方式 | 许可与使用限制 |
| --- | --- | --- | --- | --- |
| 13 | [DEMAND](https://zenodo.org/records/1227121) | 16 通道真实环境连续录音；发布 48／16 kHz 版本，选包时区分采样率 | Zenodo 按环境、采样率提供归档；随附 [DEMAND 文档](https://zenodo.org/records/1227121/files/DEMAND.pdf)说明格式及条款 | CC BY-SA 3.0。与 VoiceBank+DEMAND 噪声同源，应按源环境与原始录音隔离；不能把基准测试噪声回收到训练素材中 |
| 14 | [FSD50K](https://fsannotator.upf.edu/fsd/release/FSD50K/) | 44.1 kHz、单声道 PCM16；51,197 段、108.3 小时、200 类声音；[论文](https://arxiv.org/abs/2010.00475)给出规格及划分 | 官方配套页面指向 [Zenodo 发布记录](https://zenodo.org/records/4060432)，提供 dev／eval 音频、标签和元数据 | 逐片段采用 CC0、CC BY、CC BY-NC 或 CC Sampling+，不能以数据集整体的 CC BY 覆盖音频许可。按片段元数据筛选用途并保留署名 |
| 15 | [QUT-NOISE](https://researchdatafinder.qut.edu.au/display/n442) | 48 kHz、双声道、16-bit WAV；超过 10 小时，20 个录音 session，覆盖 CAFE／CAR／HOME／REVERB／STREET 五类场景，见[论文](https://eprints.qut.edu.au/38144/) | QUT 发布页提供 `QUT_NOISE.zip` 及五类场景分卷；[官方仓库](https://github.com/qutsaivt/QUT-NOISE)提供说明和生成代码 | 发布页列噪声为 CC BY-SA 4.0，代码为 BSD 类许可；保留随包具体条款。仅用噪声不需要 TIMIT；构建 QUT-NOISE-TIMIT 才需要 TIMIT 授权 |
| 16 | [MUSAN](https://www.openslr.org/17/) | 16 kHz WAV；总计约 109 小时，其中噪声约 6 小时，其余为语音和音乐，见[原始论文](https://arxiv.org/abs/1510.08484) | OpenSLR SLR17 提供 `musan.tar.gz` 及镜像 | OpenSLR 整体列 CC BY 4.0；各来源子目录另有逐文件许可和署名，含 Creative Commons 及美国公有领域素材。不是百小时纯噪声库，也不是原生全频带音频 |
| 17 | [ESC-50](https://github.com/karolpiczak/ESC-50) | 44.1 kHz、单声道 WAV；2,000 段，每段 5 秒，50 类环境声音 | 官方 GitHub 仓库含音频、元数据、署名及 ZIP 入口 | 完整 ESC-50 为 CC BY-NC 3.0，`esc10` 标记的 ESC-10 子集为 CC BY。保留官方 5 folds；同一源录音的片段不能跨集合。部分源音频有与类别相关的带宽限制 |
| 18 | [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) | 8,732 段、每段不超过 4 秒、10 类城市声音；WAV 的采样率、位深和声道随原录音变化 | 官方页面提供下载表单及 `soundata` 下载方式 | [官方使用条款](https://urbansounddataset.weebly.com/)为 CC BY-NC 3.0，保留 `FREESOUNDCREDITS.txt`。按源录音隔离；复现分类基准须使用官方 10 folds，不重新随机划分 |

## 房间脉冲响应与混响：2 项

RIR 用来生成房间和麦克风位置造成的声学变化，不能单独替代语音或背景噪声数据集。

| # | 数据集与官方来源 | 内容与采样率 | 下载入口与方式 | 许可与使用限制 |
| --- | --- | --- | --- | --- |
| 19 | [BUT ReverbDB](https://speech.fit.vut.cz/software/but-speech-fit-reverb-database) | 发布页列 9 个房间；含实测 RIR、背景噪声及回放语音；[公开 README](http://merlin.fit.vutbr.cz/ReverbDB/read_me.txt)说明格式为单声道、16 kHz PCM16 WAV | 官方页分别提供 RIR／噪声／元数据包和 LibriSpeech 回放语音包 | 发布页列数据为 CC BY 4.0；NIST SRE／HUB5 回放语音另受 LDC 限制。README 为较早版本，使用时核对所取归档的格式与条款，不把采录设备的 48 kHz 设置当成发布采样率 |
| 20 | [OpenAIR](https://pure.york.ac.uk/portal/en/datasets/openair-the-open-acoustic-impulse-response-library/) | 多建筑与场地的实测声学响应及消声录音；采样率、声道和空间格式需按具体条目确认 | 约克大学登记页保留项目来源；音频条目见[约克托管的 OpenAIR 站点](https://www.openair.hosted.york.ac.uk/) | 按条目的 Attribution 和下载包确认许可与署名，不推定整库统一授权。若站点不可用，可通过大学登记页联系维护者；取得 RIR 后检查格式及有效带宽再用于卷积 |

## 当前项目的数据格式

当前 [ANR-Lite-48 训练入口](../crates/anr-lite/README.md)使用 VoiceBank-DEMAND 28-speaker 目录格式，不是通用数据集加载器。发布页应选择四个归档：

- `clean_trainset_28spk_wav.zip`
- `noisy_trainset_28spk_wav.zip`
- `clean_testset_wav.zip`
- `noisy_testset_wav.zip`

从仓库根目录启动时，默认数据位置为 `datasets/VoiceBank-DEMAND 28-speaker/`；可用 `--dataset` 指定其他根目录。解压后应有四个同名、不带 `.zip` 的子目录，无需文本转录。

加载器要求：

- 四个目录内的 WAV 均为非空、单声道、48,000 Hz、PCM16；只扫描目录第一层。
- noisy／clean 文件名完全对应，采样点数相同，不允许单边缺失；文件 ID 使用 `<speaker>_<utterance>.wav`，例如 `p225_001.wav`。
- 验证集说话人从训练目录中留出；官方测试目录与训练目录不得共享说话人。加载器按文件名的说话人前缀检查。

读取时以 `sample / 32768.0` 转为浮点，noisy／clean 使用相同裁剪位置，短片段尾部补零；不会重采样或分别做峰值归一化，也不会修改数据集。其他干净语音、噪声或 RIR 需要先完成任务所需的混合、对齐及格式转换，再组织成上述配对。

## 选型与数据划分

- 先按任务选择参考：DNS 提供合成素材，VoiceBank+DEMAND 提供固定配对，CHiME-4 含实录与模拟分支，Clarity 面向双耳助听器；RIR 与独立噪声不能代替干净目标语音。
- 高采样率不保证高有效带宽。16／24 kHz 音频升采样到 48 kHz 不会补回高频；44.1 kHz 来源也应记录原始带宽及重采样过程。
- VoiceBank+DEMAND 与 VCTK、DEMAND 同源，EARS-WHAM 与 WHAM! 共享噪声，DNS 也包含部分上述来源。Hi-Fi TTS、LibriTTS-R 与 DNS 的 LibriVox 来源也可能重叠；FSD50K、ESC-50、UrbanSound8K 等 Freesound 来源应检查原始 clip ID。按说话人、原始录音与场地划分，不能只按切片文件名随机分组。
- 保留官方划分及生成版本。标准测试集不参与模型选择、噪声素材回收或增益／时延拟合；另留真实设备录音评估合成场景之外的效果。

## 获取与许可

- 表内链接是发布与说明入口，不代表归档已完成下载、完整性校验或质量检查。下载后按官方校验值检查文件，并随实验记录保存数据版本、来源清单、许可和划分。
- 组合数据集按组件及片段审查授权；代码许可不替代音频许可。CC BY 要求署名，NC 限制商业用途，SA 涉及共享条件；具体适用范围以相应许可文本为准。
- DAPS 的音频复用许可、Clarity 全包各组件的授权范围，以及 OpenAIR 具体条目的格式与许可，应在使用前从随包说明或维护者取得确认。CHiME-4 与完整 WHAM! 语音混合需另行满足对应 LDC／WSJ0 条件。
