# ANR-Lite-48

ANR-Lite-48 is a trainable 48 kHz speech-enhancement model implemented in **Burn 0.22.1**. This crate provides the network, differentiable streaming audio processing, VoiceBank-DEMAND training, and checkpoint-to-WAV rendering. It includes **no pretrained weights** and does not download datasets. A randomly initialized model demonstrates execution, not denoising quality.

[English](#english) | [简体中文](#简体中文)

## English

This guide follows the public workflow from the model API through training, WAV rendering, and an objective comparison with DFN3. ANR-Lite is independent of the existing DFN3 inference backends: a Burn record is not ONNX, and the `anr` CLI has no ANR-Lite record backend.

- [Project overview](https://github.com/lona-cn/anr/blob/v0.3.0/README.md) · [DFN3 CLI and backends](https://github.com/lona-cn/anr/blob/v0.3.0/crates/anr-cli/README.md)
- [API and streaming](#api-and-streaming) · [Data requirements](#voicebank-demand-data-requirements) · [Training](#training)
- [Render a checkpoint](#render-a-checkpoint) · [Export DFN3](#export-dfn3-with-fixed-delay-calibration) · [Score and report](#score-and-report)
- [Dataset sources and terms](https://github.com/lona-cn/anr/blob/v0.3.0/datasets/DATASETS.md) · [Complete streaming example](examples/smoke.rs)

### Registry dependencies and installed tools

The registry examples below are for **after v0.3.0 is published to crates.io**; versioned repository links also require the `v0.3.0` tag. Until then, use the local-checkout commands later in this guide.

For a library consumer, add the default CPU Flex backend to your project's `Cargo.toml`:

```toml
[dependencies]
anr-lite = "0.3.0"
```

Alternatively, select only the optional wgpu backend (no CPU Flex):

```toml
[dependencies]
anr-lite = { version = "0.3.0", default-features = false, features = ["wgpu"] }
```

Use a current stable Rust toolchain. Install both tools with the default CPU backend:

```text
cargo install anr-lite --version 0.3.0 --locked --features train --bin anr-lite-train --bin anr-lite-evaluate
```

Or choose the GPU-only build instead, requiring a supported discrete GPU and drivers:

```text
cargo install anr-lite --version 0.3.0 --locked --no-default-features --features train,wgpu --bin anr-lite-train --bin anr-lite-evaluate
```

With Cargo's install directory on `PATH`, the installed binaries run from **any working directory**. Supply your own extracted dataset, real trained checkpoint and new output paths; no repository files, pretrained weights or dataset download are supplied. This PowerShell CPU example uses absolute paths: replace them with your paths, keep outputs outside the dataset, and choose output directories that do not already exist.

```powershell
$dataset = 'C:/data/VoiceBank-DEMAND 28-speaker'
$trainRun = 'C:/anr-runs/lite-train-001'
anr-lite-train --dataset $dataset --output-dir $trainRun --device cpu --epochs 50 --batch-size 8 --segment-seconds 4 --learning-rate 0.001 --seed 42 --validation-speakers p226,p287

# Run only after training completes, using its actual best.bin.
$checkpoint = Join-Path $trainRun 'best.bin'
$liteOut = 'C:/anr-runs/lite-render-001'
anr-lite-evaluate --checkpoint $checkpoint --input-dir "$dataset/noisy_testset_wav" --output-dir $liteOut --device cpu --batch-size 8
```

For the GPU-only installation, replace `--device cpu` in **both** invocations with `--device wgpu --gpu-index 0`. `anr-lite-evaluate` renders WAVs, not PESQ/STOI scores; the data requirements and quality/reporting boundaries below still apply.

The later `cargo run -p ...` / `cargo build -p ...` commands are **checkout reproduction examples**: run them from the repository root, where their relative paths are defined. The Python export/scoring scripts, pinned Python requirements and dataset catalog belong to the repository, not the installed tools or the `anr-lite` source package. Check out the matching source revision to use that complete comparison workflow. Packaged Rust examples and source files remain linked locally.

### API and streaming

Burn is pinned to **0.22.1**; its `Tensor` API is not backend-generic.

| Cargo feature | Purpose |
| --- | --- |
| `flex` | Enabled by default; Burn CPU execution |
| `wgpu` | Optional GPU backend; select it at runtime with `--device wgpu` in the training/rendering tools |
| `train` | Enables `anr-lite-train` and `anr-lite-evaluate`, optimizer support, WAV I/O, arguments, and JSON; does not select a device |

Burn's `autodiff` and `signal` features are already dependencies, not additional crate features. CPU uses `Device::flex()`. With `wgpu` enabled, use `Device::wgpu(DeviceKind::DiscreteGpu(0))` from `anr_lite::burn::tensor`; this requires a supported discrete GPU and drivers. `--gpu-index` defaults to 0 and selects Burn's discrete-GPU index, not a CPU device. Unavailable compiled backends are errors; the tools do not silently fall back to CPU.

This untrained, silent-input example illustrates shapes and alignment, not denoising quality:

```rust,no_run
use anr_lite::{AnrLite48, AudioProcessor};
use anr_lite::burn::{module::Module, tensor::{Device, Tensor}};

fn main() {
    let device = Device::flex();
    let model = AnrLite48::new(&device).valid();
    let processor = AudioProcessor::new(&device);
    let input = Tensor::<2>::zeros([1, 960], &device);
    let output = processor.forward(&model, input);
    assert_eq!(output.dims(), [1, 960]);
}
```

The signal path uses 48 kHz mono audio, a 960-point FFT and **480-sample (10 ms) hops**, with 64-band gains and five-tap complex residual filtering in the lowest 96 bins. Its known streaming delay is **1440 samples (30 ms)**: two frames of spectral lookahead plus one frame of overlap delay, excluding time spent collecting the current input hop. The zero-initialized residual head does not make the whole randomly initialized model an identity processor.

- **Offline:** `AudioProcessor::forward(&model, input)` accepts `[batch, samples]` with both dimensions nonzero. Input, processor and model must share a device. Each call creates fresh state, pads the final short hop, flushes, removes the 1440-sample delay, and returns the original length. It does not resample.
- **Continuous streaming:** call `processor.initial_state(batch)` once, then `processor.forward_frame(&model, frame, &mut state)` for each `[batch, 480]` frame. Output has the same shape but retains the delay. Batch rows are independent streams; preserve stream identity, batch size and device across calls. Do not process adjacent chunks with separate offline `forward` calls.
- **End of stream:** right-pad only the final short input hop to 480 samples. Call `processor.flush(&model, &mut state)` **once**, obtaining `[batch, 1440]`; concatenate frame outputs and flush, then select `[1440, 1440 + original_samples)`. Flush feeds three zero hops and advances state; it is not a reset and does not belong between ordinary chunks. Start a new recording with fresh state.
- Audio, EMA, spectral history and recurrent state retain cross-frame computation graphs. `AudioState::detach()`, `ModelState::detach()` and `SpectralState::detach()` stop historical gradients without clearing values; use them only for deliberate truncated backpropagation.
- The network-only `AnrLite48::forward_frame` accepts coarse `[batch, 64]` and fine `[batch, 3, 96]` features with an explicit input/output `ModelState`. Invalid shapes or devices panic.

The [streaming example](examples/smoke.rs) implements the padding/flush/crop sequence and checks whole-waveform agreement and input gradients; it performs no optimizer update:

```text
cargo run -p anr-lite --locked --example smoke
cargo run -p anr-lite --locked --features wgpu --example smoke -- --wgpu
```

For custom training use `device.clone().autodiff()` and `model.train()`; use `model.valid()` for inference with frozen BatchNorm statistics. A record contains model parameters and normalization state, not live stream state. To load a trained record:

```rust,no_run
use std::{error::Error, path::Path};
use anr_lite::AnrLite48;
use anr_lite::burn::{module::Module, tensor::Device};

fn load_model(path: &Path, device: &Device) -> Result<AnrLite48, Box<dyn Error>> {
    Ok(AnrLite48::new(device).try_load_file(path)?.valid())
}
```

Transforms, features, network and filtering run on the chosen Burn device. This is not an allocation-free audio callback implementation; first-call timing is not steady-state performance.

### VoiceBank-DEMAND data requirements

Check the [dataset sources and usage terms](https://github.com/lona-cn/anr/blob/v0.3.0/datasets/DATASETS.md), then extract the **28-speaker, 48 kHz** release. No transcription or FFmpeg is needed. The trainer neither downloads nor modifies the corpus. Its default dataset parent is:

```text
datasets/VoiceBank-DEMAND 28-speaker/
├── clean_trainset_28spk_wav/
├── noisy_trainset_28spk_wav/
├── clean_testset_wav/
└── noisy_testset_wav/
```

The [loader](src/bin/train/data.rs) requires:

1. Exactly matching clean/noisy filenames in each split, with equal sample counts for each pair and no unpaired extra files on either side.
2. Nonempty **mono, 48,000 Hz, integer PCM16 WAVs**. Stereo, 16 kHz or float WAVs are not converted automatically. WAV entries must be regular files.
3. Names of the form `<speaker>_<utterance>.wav`, with both parts nonempty; the speaker is the part before the first underscore.
4. PCM16 decoding as `f32 / 32768`, without separate clean/noisy normalization. Paired random crops share a start position; short utterances are right-padded and loss accounts for valid length.
5. Speaker-held-out validation from the training directories, by default `p226,p287`. A comma-separated `--validation-speakers` list must contain at least one speaker, all present, while leaving a nonempty training split.
6. No speaker overlap between training directories and official test directories. Test pairs are discovered and checked at startup, but never used for gradients or best-checkpoint selection. Only `--evaluate-test` computes final test loss.

These checks establish format and pairing, not official corpus provenance or completeness: a small custom corpus can pass them. Preserve the source, full splits and configuration for an experiment. Full official test scoring has additional rules below.

### Training

Choose **one** of the following PowerShell commands. Each creates a unique new output path outside the dataset; keep `$trainRun` for subsequent rendering.

Training defaults are `--batch-size 160 --segment-seconds 4`. The CPU examples explicitly use batch size 8. These defaults can exceed GPU memory; choose a smaller `--batch-size` explicitly if needed. The trainer does not automatically reduce the batch or fall back to CPU.

```powershell
$dataset = 'datasets/VoiceBank-DEMAND 28-speaker'
$trainRun = Join-Path 'target/anr-lite-training' ('en-cpu-' + [guid]::NewGuid().ToString('N'))
cargo run -p anr-lite --release --locked --features train --bin anr-lite-train -- --dataset $dataset --output-dir $trainRun --device cpu --epochs 50 --batch-size 8 --segment-seconds 4 --learning-rate 0.001 --seed 42 --validation-speakers p226,p287
```

GPU-only build with explicit discrete-GPU selection:

```powershell
$dataset = 'datasets/VoiceBank-DEMAND 28-speaker'
$trainRun = Join-Path 'target/anr-lite-training' ('en-wgpu-' + [guid]::NewGuid().ToString('N'))
cargo run -p anr-lite --release --locked --no-default-features --features train,wgpu --bin anr-lite-train -- --dataset $dataset --output-dir $trainRun --device wgpu --gpu-index 0 --epochs 50 --batch-size 160 --segment-seconds 4 --learning-rate 0.001 --seed 42 --validation-speakers p226,p287
```

Each epoch shuffles files and takes a paired random segment from each file. Training uses AdamW (weight decay `1e-4`, gradient norm clipping 5), compressed-magnitude and compressed-complex spectral MSE (exponent 0.3, power epsilon `1e-8`), plus waveform L1 weighted by 0.1. Loss analysis uses Hann windows, FFT 960 and hop 480, excluding pure padding and weighting partial frames by valid Hann energy. Validation uses `valid()` and complete utterances in batches of at most eight; loss is weighted by valid sample count, unlike the equal-utterance quality metrics below.

For a parameter-updating smoke run, use another new directory. It is not full training or evidence of final quality:

```powershell
$smokeRun = Join-Path 'target/anr-lite-training' ('en-smoke-' + [guid]::NewGuid().ToString('N'))
cargo run -p anr-lite --locked --features train --bin anr-lite-train -- --dataset $dataset --output-dir $smokeRun --device cpu --epochs 1 --batch-size 2 --segment-seconds 0.04 --max-steps 2 --max-validation-items 1 --evaluate-test --max-test-items 1
```

`--max-steps` still performs validation and checkpoint saving for the current, possibly partial, epoch. `--max-validation-items` selects the first N sorted validation files; best selection then applies only to that subset. `--evaluate-test` reloads `best.bin` after training to compute the same loss on test data; `--max-test-items` requires it. Limits must be positive. Omit these limits for full runs. Epochs, batch size, segment duration and learning rate must also be positive (duration/rate finite); `--log-every` defaults to 10.

Outputs are `config.json`, `splits.json`, `metrics.jsonl`, `summary.json`, and:

- **`best.bin`:** updated only on a strictly lower validation loss, never selected by test loss.
- **`last.bin`:** saved after the most recently completed validation, not after every optimization step.

Existing output directories are rejected. If omitted, `--output-dir` gets a new `target/anr-lite-training/<nanoseconds>-<pid>` path. Checkpoints contain parameters and BatchNorm state, **not optimizer state, RNG state or data position**; there is no `--resume` and no exact-resume support. Keep the recorded split, step count, partial-epoch status and validation limits with any result.

### Render a checkpoint

`anr-lite-evaluate` renders WAVs; it does **not** calculate PESQ or STOI. Use an actual completed training run's `best.bin`, not an untrained model. In a new shell, set `$trainRun` to that real directory first.

```powershell
$dataset = 'datasets/VoiceBank-DEMAND 28-speaker'
$checkpoint = Join-Path $trainRun 'best.bin'
$qualityRun = Join-Path 'target/anr-lite-quality' ('en-' + [guid]::NewGuid().ToString('N'))
$liteOut = Join-Path $qualityRun 'anr-lite'
$dfnOut = Join-Path $qualityRun 'dfn3'
$scoresOut = Join-Path $qualityRun 'scores'

cargo run -p anr-lite --release --locked --no-default-features --features train,wgpu --bin anr-lite-evaluate -- --checkpoint $checkpoint --input-dir "$dataset/noisy_testset_wav" --output-dir $liteOut --device wgpu --gpu-index 0 --batch-size 8
```

For CPU rendering, retain default `flex`, use `--features train` instead of the GPU-only feature flags, and pass `--device cpu`. The renderer reads sorted mono 48 kHz PCM16 WAVs, loads the record in `valid()` mode, pads each batch to its longest utterance with independent per-row stream state, then trims each output to its original length. Outputs are same-name **48 kHz mono float32 WAVs**, already flushed and delay-compensated, with no clipping, normalization or resampling.

`--device` defaults to cpu, `--gpu-index` to 0, and `--batch-size` to 8; optional positive `--limit` is for smoke rendering. The output directory must not exist. `manifest.json` records the actual device, checkpoint size, lengths, padding and timing. `render_seconds` sums batch upload/inference/completed-download wall time, excluding WAV I/O and checkpoint loading; it is not a low-latency streaming benchmark. Failures can leave partial audio: use a new output path for another attempt.

### Export DFN3 with fixed-delay calibration

[The DFN3 exporter](https://github.com/lona-cn/anr/blob/v0.3.0/tools/export_dfn3_voicebank.py) runs the real `anr` executable and keeps scratch/logs outside the corpus. Prepare:

- The trained Lite record and the same official clean/noisy test corpus.
- An isolated **Python 3.12** environment with the [pinned quality dependencies](https://github.com/lona-cn/anr/blob/v0.3.0/tools/requirements-anr-lite-quality.txt). Building `pesq` from source on Windows requires a compatible C/C++ build toolchain.
- A working `anr` build for the chosen backend, with its drivers/runtime libraries. The example uses native DFN3 `wgpu`, not ORT.
- The fixed `models/dfn3.onnx`, SHA-256 `fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9`. The CLI build below does not automatically download it; see [the project overview](https://github.com/lona-cn/anr/blob/v0.3.0/README.md) for source and licensing.

Windows environment preparation (or set `$py` to an already prepared interpreter):

```powershell
$qualityEnv = Join-Path 'target/anr-lite-quality-envs' ('en-' + [guid]::NewGuid().ToString('N'))
python -m venv $qualityEnv
$py = Join-Path $qualityEnv 'Scripts/python.exe'
& $py -m pip install numpy==1.26.4 Cython==3.1.2 wheel==0.45.1 setuptools==80.9.0
& $py -m pip install --no-build-isolation -r tools/requirements-anr-lite-quality.txt
```

On Unix the virtual-environment interpreter is `bin/python`; the Python tools do not require PowerShell. Installation is a separate setup step: the exporter and scorer do not install packages or access the network.

Build the CLI and locate Cargo's configured target directory rather than assuming `target/release`:

```powershell
cargo build -p anr-cli --release --locked --no-default-features --features wgpu --bin anr
$cargoTarget = (cargo metadata --format-version 1 --no-deps | ConvertFrom-Json).target_directory
$anr = Join-Path $cargoTarget 'release/anr.exe'
& $py tools/export_dfn3_voicebank.py --executable $anr --model models/dfn3.onnx --input-dir "$dataset/noisy_testset_wav" --output-dir $dfnOut --backend wgpu --strength 0
```

Use the target-triple-specific executable when explicitly cross-compiling; a native Unix executable is named `anr`. The exporter defaults to **cuda**, so keep the explicit `--backend wgpu` here. `--provider` applies only to ORT (omission is recorded as cpu); ORT needs the appropriate build/runtime from the [CLI guide](https://github.com/lona-cn/anr/blob/v0.3.0/crates/anr-cli/README.md). `--strength` must be 0; positive `--limit` is smoke-only. Every export needs a new output directory.

**Strength 0 means pure wet, not bypass.** CLI strength maps to rounded attenuation dB; a tiny positive value may still round to 0 dB and cannot provide an identity reference. The exporter runs an independent random probe at strength 0 and at `1.6666667` (a 1 dB limit), recovers dry from `limited = (1-alpha) * wet + alpha * dry`, and verifies the full unscaled reconstruction. Relative RMS and absolute peak errors must both be at most `2e-5`.

Shader history indexing defines formal wet delay as the verified dry delay **minus one 480-sample hop**. An independent 8–18 kHz probe verifies that fixed delay; it does not define or optimize it. Low-frequency complex filtering can change phase, so full-band correlation peaks are not used to choose a new lag per utterance. Calibration failure aborts the export.

Every noisy test file is padded in scratch for the tail, processed and cropped to its original length with the same fixed delay. The tool never reads test clean, fits per-utterance lag/gain, clips or normalizes. Outputs include aligned WAVs, `manifest.json`, `logs/` and `scratch/`; the manifest records executable/model hashes, actual commands, probe calibration and per-file details.

### Score and report

[The scorer](https://github.com/lona-cn/anr/blob/v0.3.0/tools/evaluate_voicebank.py) compares the noisy baseline and enhanced directories with clean:

```powershell
& $py tools/evaluate_voicebank.py --clean-dir "$dataset/clean_testset_wav" --noisy-dir "$dataset/noisy_testset_wav" --enhanced "anr-lite=$liteOut" --enhanced "dfn3=$dfnOut" --output-dir $scoresOut --workers 8 --bootstrap-samples 2000 --seed 42
```

`--enhanced NAME=DIR` is repeatable and required at least once; `clean` and `noisy` are reserved names. Use exactly `anr-lite` and `dfn3` for the built-in paired model difference. Output must be a new directory. Workers (default 8) and bootstrap samples (default 2000) must be positive; seed (default 42) must be nonnegative.

- **Official full test:** omit `--limit`; clean must contain **824 WAVs**, and noisy plus every enhanced directory must have exactly the same complete filename set, with no missing or extra WAVs. Count/set checks do not authenticate provenance: verify that this is the official test corpus.
- Scored audio must be nonempty, finite, mono 48 kHz and exactly the paired clean length. PCM and float WAVs are accepted; do not quantize enhanced float output merely for scoring.
- **Smoke:** `--limit N` selects the first N sorted clean names, but checks complete filename-set equality **before** selection. Limited render outputs do not match full clean/noisy directories. Either render everything and limit only scoring, or create new matching clean/noisy subset directories and use that subset for both renderers and the scorer; never move or rewrite the official corpus.
- Any explicit scoring limit is labeled `smoke_sorted_prefix`, even if it happens to include 824 files; it is not `official_test_full_824`. Renderers do not enforce the official count; the scorer's input requirements, scope and counts determine the reported coverage.

| Metric | Definition |
| --- | --- |
| `wb_pesq` | WB-PESQ after converting clean/estimate to 16 kHz using `scipy.signal.resample_poly(up=1, down=3)` |
| `stoi` | The same 16 kHz conversion, `pystoi.stoi(..., extended=False)`; not ESTOI |
| `si_sdr_db` | At 48 kHz, independently demean signals and compute projected/residual energy dB |
| `snr_db` | At 48 kHz, `10*log10(sum(clean²)/sum((estimate-clean)²))`; no demeaning or gain/lag fitting |

All four are higher-is-better. Summaries are **equal-utterance arithmetic means**, not duration-weighted means. There is no extra level normalization or reference-fitted alignment. Invalid/nonfinite audio, undefined zero-energy ratios, PESQ failures and STOI warnings are errors, not floored scores or silently discarded samples.

Only after all selected files succeed does the scorer create its output directory:

- `per_file.csv` / `per_file.json`: filename, model, length, four metrics, peak, RMS and `abs(sample) >= 1` counts. Full-scale exceedance alone does not prove clipping.
- `paired_differences.csv`: per-utterance ANR-Lite minus DFN3 differences.
- `summary.json`: scope, source/evaluated/not-evaluated counts, per-model means, dependency versions, metric conventions and paired-mean **95% percentile confidence intervals**, bootstrapped by resampling utterances with replacement.

Keep training configuration/splits, selected checkpoint, both rendering manifests and scoring summary together. Distinguish full training, limited steps, limited validation, full test and smoke results. Training loss is not an audio-quality metric; objective scores and utterance bootstrap over two test speakers establish neither subjective quality nor generalization to new speakers/domains. DFN3 pretraining data and test overlap have not been audited here, so a difference is not a controlled architecture-only comparison. This guide supplies no unmeasured speed or quality results.

## 简体中文

本指南覆盖从模型 API、数据准备到训练、音频导出和 DFN3 客观对比的完整流程。ANR-Lite 与项目现有的 DFN3 推理后端相互独立：Burn record 不是 ONNX，`anr` CLI 不会选择或自动加载 ANR-Lite 权重。

- [项目入口](https://github.com/lona-cn/anr/blob/v0.3.0/README.md)
- [DFN3 CLI 与后端配置](https://github.com/lona-cn/anr/blob/v0.3.0/crates/anr-cli/README.md)
- [数据集来源与使用条件](https://github.com/lona-cn/anr/blob/v0.3.0/datasets/DATASETS.md)
- [完整流式示例](examples/smoke.rs)

### crates.io 依赖与已安装工具

以下 registry 示例供 **v0.3.0 发布到 crates.io 后**使用；带版本的仓库链接也需要 `v0.3.0` tag。发布前可使用本指南后面的本地 checkout 命令。

库调用方在自己项目的 `Cargo.toml` 中添加依赖，默认启用 CPU Flex 后端：

```toml
[dependencies]
anr-lite = "0.3.0"
```

也可改为仅启用可选的 wgpu 后端（不编入 CPU Flex）：

```toml
[dependencies]
anr-lite = { version = "0.3.0", default-features = false, features = ["wgpu"] }
```

使用当前 stable Rust。以下命令安装两个工具，默认使用 CPU 后端：

```text
cargo install anr-lite --version 0.3.0 --locked --features train --bin anr-lite-train --bin anr-lite-evaluate
```

如果只使用 GPU，改用以下安装命令，需要受支持的独立 GPU 和驱动：

```text
cargo install anr-lite --version 0.3.0 --locked --no-default-features --features train,wgpu --bin anr-lite-train --bin anr-lite-evaluate
```

将 Cargo 安装目录加入 `PATH` 后，已安装的程序可从**任意工作目录**运行。须自行提供已解压数据集、实际训练得到的 checkpoint 和新的输出路径；安装不附仓库文件、预训练权重或数据集下载。下面是使用绝对路径的 PowerShell CPU 示例：替换为自己的路径，输出须位于数据集之外，且输出目录不得已存在。

```powershell
$dataset = 'C:/data/VoiceBank-DEMAND 28-speaker'
$trainRun = 'C:/anr-runs/lite-train-001'
anr-lite-train --dataset $dataset --output-dir $trainRun --device cpu --epochs 50 --batch-size 8 --segment-seconds 4 --learning-rate 0.001 --seed 42 --validation-speakers p226,p287

# 仅在训练完成后执行，使用该次训练实际生成的 best.bin。
$checkpoint = Join-Path $trainRun 'best.bin'
$liteOut = 'C:/anr-runs/lite-render-001'
anr-lite-evaluate --checkpoint $checkpoint --input-dir "$dataset/noisy_testset_wav" --output-dir $liteOut --device cpu --batch-size 8
```

若选择 GPU-only 安装，将**两个**调用中的 `--device cpu` 替换为 `--device wgpu --gpu-index 0`。`anr-lite-evaluate` 只渲染 WAV，不计算 PESQ/STOI；下文的数据要求和质量/报告边界仍然适用。

后面的 `cargo run -p ...` / `cargo build -p ...` 是 **checkout 复现实例**，从仓库根目录执行，相对路径也以此为基准。Python 导出/评分脚本、固定 Python 依赖文件及数据集目录说明属于仓库，不随已安装工具或 `anr-lite` 源码包提供；使用完整对比流程须检出对应源码版本。已打包的 Rust 示例和源文件仍保留包内相对链接。

### 架构与信号路径

| 项目 | 实现 |
| --- | --- |
| 波形输入 | 48,000 Hz，单声道；批处理张量 `[batch, samples]` |
| 流式 hop | `FRAME_SIZE = 480`，即 10 ms |
| 分析/合成 | 960 点实数 FFT，481 个频点，50% overlap，Vorbis 窗 |
| 粗粒度特征 | 64-band ERB 功率谱；dB 特征减去 EMA，再除以 40；EMA 更新系数 0.01 |
| 细粒度特征 | 最低 96 个频点的 log magnitude、幅度归一化 real/imaginary，形状 `[batch, 3, 96]` |
| 编码器 | 两支 16 通道网络；coarse 频率宽度 `64 → 32 → 16`，fine 为 `96 → 48` |
| 共享时序网络 | 八分组 `1024 → 128` 投影，两层 128 维、四分组 GRU；每层后以全连接层进行跨组混合，并加 residual |
| 解码器 | 八分组 `128 → 1024` 投影，nearest-neighbor 上采样、encoder skip addition、depthwise/pointwise 卷积、BatchNorm、PReLU |
| 输出头 | sigmoid 全带增益 `[batch, 64]`；tanh 复数残差系数 `[batch, 5, 96, 2]`，最后一维为 real/imaginary |
| 频谱增强 | 增益作用于目标帧 `t-2`；低频另叠加当前帧至 `t-4` 的五 tap 复数残差，tap 0 对应当前频谱 |
| 残差范围 | bin 0–95；bin 80–95 使用 cosine taper，bin 95 残差为零；更高频率仅走增益路径 |
| 流式延迟 | `AUDIO_DELAY_SAMPLES = 1440`，即 30 ms；两帧 spectral lookahead 加一帧 overlap 延迟，不含收集当前输入 hop 的时间 |

残差输出头以零初始化；其余可训练层按各层初始化器创建。这不使整个模型成为 identity，也不提供可用的预训练降噪器。GRU 状态保存的是两个独立的 unmixed hidden states，而不是跨组混合后的 residual activation。

`Fft960` 将 Burn 原生 64 点 FFT 与固定 radix-15 阶段组合。正变换不归一化，逆变换归一化为 `1/960`。ERB 分析/插值使用稀疏映射，每个频点最多连接两个 band，不构造稠密频率矩阵。变换、特征、网络和滤波在所选 Burn device 上执行；模型不是无分配音频 callback 实现，首次调用耗时也不是稳态性能指标。

实现位置：[audio.rs](src/audio.rs)、[spectral.rs](src/spectral.rs)、[network.rs](src/network.rs)、[layers.rs](src/layers.rs)、[fft.rs](src/fft.rs)。

### Features 与 Burn API

默认依赖为精确版本 `burn = 0.22.1`；示例使用该版本的 `Device` 与非 backend-generic `Tensor` API。

| Feature | 用途 |
| --- | --- |
| `flex` | 默认启用，Burn CPU 执行后端 |
| `wgpu` | 可选 Burn wgpu 执行后端；训练/导出仍须显式传 `--device wgpu` |
| `train` | 构建 `anr-lite-train` 和 `anr-lite-evaluate`，启用优化器、WAV、参数与 JSON 支持；不自动选择设备 |

`autodiff` 与 `signal` 是本 crate 的 Burn 依赖特性，不需要另加同名 crate feature。CPU 设备为 `Device::flex()`；GPU 为 `Device::wgpu(DeviceKind::DiscreteGpu(index))`，需要相应 feature、独立 GPU 和可用驱动。训练和导出程序在未编入所选后端时返回错误，不自动改选 CPU。`--gpu-index` 默认 0，只用于 `--device wgpu`；它是 Burn 的独立 GPU 序号。

#### 整段波形

下面使用未训练模型与静音，说明形状和对齐 API，不作为音质示例：

```rust,no_run
use anr_lite::{AnrLite48, AudioProcessor};
use anr_lite::burn::{module::Module, tensor::{Device, Tensor}};

fn main() {
    let device = Device::flex();
    let model = AnrLite48::new(&device).valid();
    let processor = AudioProcessor::new(&device);
    let input = Tensor::<2>::zeros([1, 960], &device);
    let output = processor.forward(&model, input);
    assert_eq!(output.dims(), [1, 960]);
}
```

GPU 调用只替换设备创建，且为 crate 启用 `wgpu`：

```rust,no_run
use anr_lite::AnrLite48;
use anr_lite::burn::{module::Module, tensor::{Device, DeviceKind}};

fn main() {
    let device = Device::wgpu(DeviceKind::DiscreteGpu(0));
    let _model = AnrLite48::new(&device).valid();
}
```

#### 状态、尾部与对齐

- `AudioProcessor::forward(&model, input)` 要求 batch 和 samples 均非零，且 input、model、processor 位于同一设备。每次调用都新建 stream state，补齐最后一个短 hop、flush，并删除已知延迟，返回与输入等长的波形。它不重采样。
- 连续录音使用 `initial_state(batch)`，再反复调用 `forward_frame(&model, frame, &mut state)`。输入、输出均为 `[batch, 480]`；同一 batch 的各行是独立流，跨调用必须保持流身份、batch 大小和设备一致。不要把相邻片段分别交给 `forward`。
- `forward_frame` 的输出保留 1440 个采样延迟。结束时将最后不足 480 个采样的输入右侧补零，再调用**一次** `flush(&model, &mut state)`，得到 `[batch, 1440]`。拼接逐帧输出与 flush 后，取区间 `[1440, 1440 + original_samples)` 即为对齐结果。
- `flush` 会喂入三个零 hop 并推进状态，不是 reset，不应放在相邻普通 chunk 之间。切换录音时重新调用 `initial_state`。
- 波形、EMA、频谱历史和循环网络状态保留跨帧计算图。`AudioState::detach()`、`ModelState::detach()`、`SpectralState::detach()` 只切断历史梯度，不清除数值；仅在有意进行 truncated backpropagation 时使用。
- `AnrLite48::forward_frame` 也可单独使用：输入 coarse `[batch, 64]`、fine `[batch, 3, 96]`，显式传入并接回 `ModelState`。形状和设备不匹配会 panic。

运行已有示例可检查整段/流式路径及输入梯度，不会执行 optimizer update：

```text
cargo run -p anr-lite --locked --example smoke
cargo run -p anr-lite --locked --features wgpu --example smoke -- --wgpu
```

#### 模型模式与 record

自定义训练循环使用 `device.clone().autodiff()` 与 `model.train()`；推理使用 `model.valid()`，冻结 BatchNorm 统计。record 包含模型参数和归一化状态，不包含 `AudioState`、`ModelState` 或正在处理的录音。

```rust,no_run
use std::{error::Error, path::Path};
use anr_lite::AnrLite48;
use anr_lite::burn::{module::Module, tensor::Device};

fn load_model(path: &Path, device: &Device) -> Result<AnrLite48, Box<dyn Error>> {
    Ok(AnrLite48::new(device).try_load_file(path)?.valid())
}
```

### VoiceBank-DEMAND 数据要求

先按[数据集目录](https://github.com/lona-cn/anr/blob/v0.3.0/datasets/DATASETS.md)核对来源和使用条件，再解压 28-speaker、48 kHz 版本。训练器不下载、不修改数据，不需要文本转录，也不依赖 FFmpeg。默认 `--dataset` 为 `datasets/VoiceBank-DEMAND 28-speaker`，目录必须为：

```text
datasets/VoiceBank-DEMAND 28-speaker/
├── clean_trainset_28spk_wav/
├── noisy_trainset_28spk_wav/
├── clean_testset_wav/
└── noisy_testset_wav/
```

[数据加载器](src/bin/train/data.rs)要求：

1. 每个 noisy WAV 有且仅有同名 clean WAV，反向也不能多出文件；两者采样数完全相同。
2. WAV 非空、**单声道、48,000 Hz、整数 PCM16**。16 kHz、立体声或 float WAV 不能直接训练；加载器不自动转换。
3. 文件名为 `<speaker>_<utterance>.wav`，两部分均非空；speaker 取第一个下划线之前的部分。WAV 条目须为普通文件，配对按完整文件名匹配。
4. PCM16 解码为 `f32 / 32768`，不分别归一化 noisy/clean。配对裁剪使用同一起点；短句右侧补零，loss 按有效长度处理。
5. 验证集只从训练目录按 speaker 留出，默认 `p226,p287`。`--validation-speakers` 可接受逗号分隔的名单；必须至少一位、全部存在，且留下非空训练集。
6. 训练目录与官方测试目录不能含同一 speaker。测试配对会在启动时被发现和检查，但不参与梯度或 best checkpoint 选择。只有 `--evaluate-test` 才在训练结束后计算测试损失。

加载器检查配对和格式，但不以官方完整语料数量认证数据来源；小型自建 corpus 也可能满足格式。正式实验需另外保留来源、完整 split 和训练配置。官方测试集完整性由后面的评分规则约束。

### 训练

下面 PowerShell 示例为每次训练生成新路径。CPU 与 GPU 命令是**二选一**；完成所选命令后，`$trainRun` 可用于后续导出。所有输出位于数据集之外。

训练默认使用 `--batch-size 160 --segment-seconds 4`。CPU 示例显式使用 batch size 8。默认配置可能超出 GPU 显存；显存不足时须手动减小 `--batch-size`，训练器不会自动减小 batch 或回退到 CPU。

```powershell
$dataset = 'datasets/VoiceBank-DEMAND 28-speaker'
$trainRun = Join-Path 'target/anr-lite-training' ([guid]::NewGuid().ToString('N'))

# CPU，保留默认 flex feature
cargo run -p anr-lite --release --locked --features train --bin anr-lite-train -- --dataset $dataset --output-dir $trainRun --device cpu --epochs 50 --batch-size 8 --segment-seconds 4 --learning-rate 0.001 --seed 42 --validation-speakers p226,p287
```

GPU-only 构建及明确的独立 GPU 选择：

```powershell
$dataset = 'datasets/VoiceBank-DEMAND 28-speaker'
$trainRun = Join-Path 'target/anr-lite-training' ([guid]::NewGuid().ToString('N'))
cargo run -p anr-lite --release --locked --no-default-features --features train,wgpu --bin anr-lite-train -- --dataset $dataset --output-dir $trainRun --device wgpu --gpu-index 0 --epochs 50 --batch-size 160 --segment-seconds 4 --learning-rate 0.001 --seed 42 --validation-speakers p226,p287
```

训练器每轮打乱文件顺序，对每个文件随机取配对片段。默认 AdamW 的 weight decay 为 `1e-4`，梯度 norm clipping 为 5。损失是压缩幅度谱 MSE、压缩复数谱 MSE（压缩指数 0.3，power epsilon `1e-8`）和权重 0.1 的波形 L1；loss 分析使用 Hann 窗、FFT 960、hop 480。纯 padding 不计入 loss，部分有效帧按有效 Hann 能量加权。

验证使用 `valid()`，每批最多八句完整语音；计算 loss 前移除每句 padding，最终按有效采样数加权。该 loss 与后面的按句平均音质指标不是同一种统计量。

#### 参数参考

| 参数 | 默认值 / 行为 |
| --- | --- |
| `--dataset PATH` | 上述四目录的父目录 |
| `--output-dir PATH` | 省略时新建 `target/anr-lite-training/<nanoseconds>-<pid>`；指定路径也必须不存在 |
| `--device cpu\|wgpu` | `cpu`；须有对应编译 feature |
| `--gpu-index N` | `0`，仅 wgpu |
| `--epochs N` | `50`，必须为正 |
| `--batch-size N` | `160`，必须为正 |
| `--segment-seconds S` | `4.0`，有限正数，换算后采样数在 `1..=u32::MAX` |
| `--learning-rate RATE` | `0.001`，有限正数 |
| `--seed N` | `42`，用于设备初始化、文件顺序与裁剪 |
| `--validation-speakers LIST` | `p226,p287` |
| `--max-steps N` | 省略时不限；正数，达到后仍执行当前轮验证和保存，可能是 partial epoch |
| `--max-validation-items N` | 省略时验证全部留出文件；限制时只取排序后的前 N 项 |
| `--evaluate-test` | 默认关闭；训练结束后重新加载 `best.bin`，计算官方测试集上的相同 loss |
| `--max-test-items N` | 需同时传 `--evaluate-test`；正数，只取前 N 项 |
| `--log-every N` | `10`，必须为正；第一步也输出 loss |

短跑检查使用新的输出路径；它确实更新参数，但不能代表完整训练或最终质量：

```powershell
$smokeRun = Join-Path 'target/anr-lite-training' ('smoke-' + [guid]::NewGuid().ToString('N'))
cargo run -p anr-lite --locked --features train --bin anr-lite-train -- --dataset $dataset --output-dir $smokeRun --device cpu --epochs 1 --batch-size 2 --segment-seconds 0.04 --max-steps 2 --max-validation-items 1 --evaluate-test --max-test-items 1
```

#### 输出与 checkpoint 边界

| 文件 | 内容 |
| --- | --- |
| `config.json` | 参数、解析后的 dataset/device、loss 与 optimizer 配置 |
| `splits.json` | train/validation/test 的文件、speaker、时长与数量 |
| `metrics.jsonl` | 每步训练、每轮验证及可选最终测试记录 |
| `summary.json` | 步数、完整轮数、partial epoch、best 位置和实际评估数量 |
| `last.bin` | 最近完成验证后保存的模型，不是每个 optimization step 都保存 |
| `best.bin` | 验证 loss 严格下降时更新的模型；不以测试 loss 选模 |

已有输出目录会被拒绝，不能以旧目录继续运行。checkpoint 仅保存参数与 BatchNorm 状态，**不保存 optimizer、RNG 或数据迭代位置，也没有 `--resume` 参数**；加载 record 推理与精确续训是不同能力。若使用限量验证，`best.bin` 也是按该限量验证集选择的，报告中应保留这一条件。

### 从 checkpoint 导出对齐音频

`anr-lite-evaluate` 只渲染音频，不计算 PESQ/STOI。它按文件名排序读取单声道 48 kHz PCM16 WAV，在 `valid()` 模式加载实际 record；每批按最长语句补零、各行使用独立新状态，再截回原始长度。输出同名 **48 kHz mono float32 WAV**，不剪裁、不归一化、不重采样，已包含 flush 与延迟补偿。

以下示例使用上面已完成训练的 `$trainRun`。若另开 shell，先将 `$trainRun` 设置为实际训练目录；不要用随机模型或未完成的目录替代 `best.bin`。

```powershell
$dataset = 'datasets/VoiceBank-DEMAND 28-speaker'
$checkpoint = Join-Path $trainRun 'best.bin'
$qualityRun = Join-Path 'target/anr-lite-quality' ([guid]::NewGuid().ToString('N'))
$liteOut = Join-Path $qualityRun 'anr-lite'
$dfnOut = Join-Path $qualityRun 'dfn3'
$scoresOut = Join-Path $qualityRun 'scores'

cargo run -p anr-lite --release --locked --no-default-features --features train,wgpu --bin anr-lite-evaluate -- --checkpoint $checkpoint --input-dir "$dataset/noisy_testset_wav" --output-dir $liteOut --device wgpu --gpu-index 0 --batch-size 8
```

CPU 导出改用 `--features train`（保留默认 `flex`），并传 `--device cpu`。导出器的全部参数为 `--checkpoint`、`--input-dir`、`--output-dir`、`--device`（默认 cpu）、`--gpu-index`（默认 0）、`--batch-size`（默认 8）和可选正数 `--limit`。不存在的后端和无法加载的 checkpoint 都是错误。

输出目录必须为新目录。完成后写入 `manifest.json`，记录实际设备、checkpoint 大小、输入/输出长度、batch padding 和耗时。`render_seconds` 为各 batch 上传、推理及完成下载的 wall time 总和，不包含 WAV I/O 或 checkpoint 加载；不能直接当作低延迟流式 benchmark。失败可能留下部分音频，新一次运行应另选新路径，不能把残留文件当作完整结果。

### DFN3 固定延迟校准与导出

[export_dfn3_voicebank.py](https://github.com/lona-cn/anr/blob/v0.3.0/tools/export_dfn3_voicebank.py)调用实际 `anr` 可执行文件，在数据集外创建 scratch 和日志。先准备：

- 已训练的 ANR-Lite checkpoint，以及同一份官方 test clean/noisy 数据；
- Python 3.12 独立环境及[固定依赖](https://github.com/lona-cn/anr/blob/v0.3.0/tools/requirements-anr-lite-quality.txt)；Windows 从源码构建 `pesq` 时需要匹配的 C/C++ 编译工具链；
- 已构建并可运行所选后端的 `anr`，以及驱动/运行库；下例明确使用原生 DFN3 `wgpu`，不需要 ORT；
- 项目固定的 `models/dfn3.onnx`，SHA-256 必须为 `fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9`。这个 `--no-default-features` CLI 构建不会自动下载模型；来源和许可见[项目说明](https://github.com/lona-cn/anr/blob/v0.3.0/README.md)。

为质量工具新建隔离环境的 Windows 示例；已有合适环境可直接设置 `$py`，不必重复安装：

```powershell
$qualityEnv = Join-Path 'target/anr-lite-quality-envs' ([guid]::NewGuid().ToString('N'))
python -m venv $qualityEnv
$py = Join-Path $qualityEnv 'Scripts/python.exe'
& $py -m pip install numpy==1.26.4 Cython==3.1.2 wheel==0.45.1 setuptools==80.9.0
& $py -m pip install --no-build-isolation -r tools/requirements-anr-lite-quality.txt
```

Unix 环境的虚拟环境解释器位于 `bin/python`；评分脚本不限定 PowerShell。依赖安装是一次性的环境准备，导出器与评分器自身不会安装依赖或联网。

构建 DFN3 CLI，再从 Cargo metadata 获取可执行文件位置，避免假定工作区使用 `target/release`：

```powershell
cargo build -p anr-cli --release --locked --no-default-features --features wgpu --bin anr
$cargoTarget = (cargo metadata --format-version 1 --no-deps | ConvertFrom-Json).target_directory
$anr = Join-Path $cargoTarget 'release/anr.exe'
& $py tools/export_dfn3_voicebank.py --executable $anr --model models/dfn3.onnx --input-dir "$dataset/noisy_testset_wav" --output-dir $dfnOut --backend wgpu --strength 0
```

若显式交叉编译，需使用对应 target triple 下的程序；Unix 原生程序名为 `anr`。导出器参数为必填 `--executable`、`--model`、`--input-dir`、`--output-dir`，以及 `--backend`（默认 **cuda**，本例明确覆盖）、`--provider`（仅 ORT）、`--strength`（必须为 0）和正数 `--limit`（仅 smoke）。使用 ORT 时还须按 [CLI 指南](https://github.com/lona-cn/anr/blob/v0.3.0/crates/anr-cli/README.md)准备对应构建和运行库；省略 provider 记录为 cpu。

#### 校准为何独立于 clean 目标

DFN3 的 `strength` 映射到取整后的 attenuation dB，**不是直接的 dB 值或旁路开关**。`strength=0` 关闭衰减限制，输出 pure wet；极小正数同样可能舍入为 0 dB，不能拿来构造 identity reference。

导出器首先生成独立随机探针，分别以 pure wet 和 `strength=1.6666667`（1 dB 限制）运行 CLI。依据实现中的 `limited = (1-alpha) * wet + alpha * dry`，反解 dry，并验证未拟合增益的整段重建；相对 RMS 和绝对峰值误差均须不超过 `2e-5`。由 shader 历史索引推导，formal wet 延迟固定为已验证 dry 延迟减去一个 480-sample hop；独立 8–18 kHz 探针只验证这一延迟，不定义或优化它。低频 complex DF 可能改变相位，不能用全带相关峰逐句重选偏移。

所有测试 noisy 先在 scratch 中补零，保留尾部所需的处理空间，再以同一个固定延迟裁回原始采样数。工具不读取 test clean，不逐句做 lag/gain fitting，也不剪裁或归一化。输出根目录含对齐 WAV、`manifest.json`、`logs/` 和 `scratch/`；manifest 记录可执行文件/模型哈希、实际命令、探针校准和各文件输入/输出信息。校准失败会中止，不能跳过后继续产生正式比较。

### 客观指标与完整评估规则

[evaluate_voicebank.py](https://github.com/lona-cn/anr/blob/v0.3.0/tools/evaluate_voicebank.py)将 noisy baseline 和一个或多个增强目录与 clean 配对评分：

```powershell
& $py tools/evaluate_voicebank.py --clean-dir "$dataset/clean_testset_wav" --noisy-dir "$dataset/noisy_testset_wav" --enhanced "anr-lite=$liteOut" --enhanced "dfn3=$dfnOut" --output-dir $scoresOut --workers 8 --bootstrap-samples 2000 --seed 42
```

必填参数为 `--clean-dir`、`--noisy-dir`、至少一个 `--enhanced NAME=DIR` 和新建的 `--output-dir`。`--enhanced` 可重复；`noisy`、`clean` 为保留名称。用准确的 `anr-lite` 与 `dfn3` 名称，才会生成内置的两模型配对差值。`--workers` 默认 8，`--bootstrap-samples` 默认 2000，均须为正；`--seed` 默认 42，须非负；正数 `--limit` 明确标记为 smoke。

#### 输入与数量

- 不传 `--limit` 时，clean 必须有完整官方测试集的 **824 个** WAV；noisy 和每个增强目录的 WAV 文件名集合必须与 clean 完全相同，不能缺文件、多文件或只评容易样本。该检查不认证语料来源，仍需自行确认使用的是官方 test。
- 所有被评文件须为非空、有限、48 kHz mono 波形，与同名 clean 采样数一致。评分器支持 PCM 或 float WAV，不要求增强输出回写为 PCM16。
- `--limit N` 选择 clean 文件名排序后的前 N 项，但**仍先检查各目录完整文件名集合相等**。只给两个渲染器添加 `--limit N`、却把完整 clean/noisy 目录交给 scorer，会因集合不一致而失败。
- 若仅限制评分开销，可先完整渲染，再给 scorer 添加 `--limit N`。若连渲染也要缩小，须另备新的、彼此同名的 clean/noisy 子集目录，再让两个渲染器和 scorer 都处理该子集；不移动或改写官方语料。
- 显式传过 `--limit` 的评分即使碰巧覆盖 824 项，也标记为 `smoke_sorted_prefix`，不能作为 `official_test_full_824` 报告。渲染器本身不检查官方 824 数量，最终以完整输入要求和 scorer 的 scope/count 为准。

#### 指标定义与报告

| 指标 | 定义 |
| --- | --- |
| `wb_pesq` | clean/estimate 均以 `scipy.signal.resample_poly(up=1, down=3)` 转为 16 kHz，调用 WB-PESQ |
| `stoi` | 使用相同 16 kHz 转换，`pystoi.stoi(..., extended=False)`，不是 ESTOI |
| `si_sdr_db` | 原始 48 kHz；分别去均值，将 estimate 投影到 clean，计算 projected/residual energy dB |
| `snr_db` | 原始 48 kHz；`10*log10(sum(clean²)/sum((estimate-clean)²))`，不去均值、不拟合增益或偏移 |

四项均为越高越好，汇总使用**逐句等权算术平均**，不是时长加权。没有指标外的音量归一化或参考拟合对齐。无效音频、非有限数、零能量导致的未定义比值、PESQ 失败或 STOI 警告都会报错，不以 epsilon floor、错误码或删样本替代。

成功完成全部选定样本后才创建评分输出目录，写入：

- `per_file.csv` / `per_file.json`：文件、模型、样本数、四项指标，以及 peak、RMS、`abs(sample) >= 1` 数量；超过满幅本身不等于已发生剪裁。
- `paired_differences.csv`：同名语句的 ANR-Lite 减 DFN3 差值。
- `summary.json`：scope、source/evaluated/not-evaluated 数量、逐模型均值、依赖版本、指标约定，以及按语句有放回 bootstrap 的 paired mean 95% percentile CI。

正式报告应关联训练配置与 split、选模 checkpoint、两份渲染 manifest 和评分 summary，区分完整训练、限步训练、限量验证、完整测试和 smoke。训练 loss 不能替代音质指标；客观指标与两位测试 speaker 的 utterance bootstrap 不建立主观听感或新 speaker/domain 泛化结论。此流程未审计 DFN3 预训练语料及其与测试集的重叠，因此模型间差异不是受控的纯架构比较。本指南不附未经测量的性能或质量结果。
