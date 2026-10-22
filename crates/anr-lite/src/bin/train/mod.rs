use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anr_lite::{
    AnrLite48, AudioProcessor, SAMPLE_RATE,
    burn::{
        module::Module,
        optim::{AdamWConfig, GradientsParams, grad_clipping::GradientClippingConfig},
        tensor::{Device, Tensor, TensorData},
    },
};
use anyhow::{Context, Result, ensure};
use rand::{RngExt, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Serialize;
use serde_json::json;

use super::{Args, Backend};
use data::{Corpus, Pair, read_segment};
use loss::EnhancementLoss;

mod data;
mod loss;

#[derive(Debug, Serialize)]
struct Evaluation {
    loss: f64,
    items: usize,
    samples: usize,
}

#[derive(Debug, Serialize)]
struct Summary {
    epoch: usize,
    epochs_completed: usize,
    steps: usize,
    partial_epoch: bool,
    train_items: usize,
    validation_items: usize,
    test_items: usize,
    validation_speakers: Vec<String>,
    validation_evaluated_items: usize,
    best_validation_loss: f64,
    best_epoch: usize,
    best_step: usize,
    best_checkpoint: &'static str,
    last_checkpoint: &'static str,
    test_loss: Option<f64>,
    test_evaluated_items: usize,
}

fn device(args: &Args) -> Result<Device> {
    match args.device {
        Backend::Cpu => {
            #[cfg(feature = "flex")]
            {
                Ok(Device::flex())
            }
            #[cfg(not(feature = "flex"))]
            {
                anyhow::bail!("CPU 训练需要 flex feature，请启用默认 features 或显式添加 flex")
            }
        }
        Backend::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                Ok(Device::wgpu(
                    anr_lite::burn::tensor::DeviceKind::DiscreteGpu(args.gpu_index),
                ))
            }
            #[cfg(not(feature = "wgpu"))]
            {
                anyhow::bail!(
                    "此程序未启用 wgpu，请使用 --features train,wgpu 重新构建，或显式选择 --device cpu"
                )
            }
        }
    }
}

fn create_output(requested: Option<&Path>) -> Result<PathBuf> {
    let path = match requested {
        Some(path) => path.to_path_buf(),
        None => {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .context("系统时间早于 UNIX_EPOCH")?;
            PathBuf::from("target/anr-lite-training").join(format!(
                "{}-{}",
                now.as_nanos(),
                std::process::id()
            ))
        }
    };
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("无法创建输出目录的父目录 {}", parent.display()))?;
    }
    fs::create_dir(&path).with_context(|| {
        format!(
            "无法新建训练目录 {}；请选择不存在的新目录，程序不会覆盖旧训练结果",
            path.display()
        )
    })?;
    Ok(path)
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut file =
        BufWriter::new(File::create(path).with_context(|| format!("无法写入 {}", path.display()))?);
    serde_json::to_writer_pretty(&mut file, value)?;
    writeln!(file)?;
    file.flush()?;
    Ok(())
}

fn split_description(pairs: &[Pair]) -> serde_json::Value {
    let speakers: BTreeSet<&str> = pairs.iter().map(|pair| pair.speaker.as_str()).collect();
    let samples: usize = pairs.iter().map(|pair| pair.samples).sum();
    json!({
        "items": pairs.len(),
        "speakers": speakers,
        "hours": samples as f64 / f64::from(SAMPLE_RATE) / 3600.0,
        "files": pairs.iter().map(|pair| &pair.id).collect::<Vec<_>>(),
    })
}

fn evaluate(
    model: &AnrLite48,
    pairs: &[Pair],
    limit: Option<usize>,
    processor: &AudioProcessor,
    loss: &EnhancementLoss,
    device: &Device,
) -> Result<Evaluation> {
    let count = limit.unwrap_or(pairs.len()).min(pairs.len());
    ensure!(count > 0, "评估数据不能为空");
    let mut weighted_loss = 0.0;
    let mut total_samples = 0usize;
    // Batch only the valid-mode waveform renderer. Keep the original
    // per-utterance loss boundaries and sample-weighted corpus reduction.
    for batch in pairs[..count].chunks(8) {
        let samples = batch.iter().map(|pair| pair.samples).max().unwrap();
        let elements = samples
            .checked_mul(batch.len())
            .context("评估 batch 大小溢出")?;
        let mut noisy = vec![0.0; elements];
        let mut clean = vec![0.0; elements];
        let mut lengths = Vec::with_capacity(batch.len());
        for (index, pair) in batch.iter().enumerate() {
            let range = index * samples..(index + 1) * samples;
            lengths.push(read_segment(
                pair,
                0,
                &mut noisy[range.clone()],
                &mut clean[range],
            )?);
        }
        let noisy = Tensor::<2>::from_data(TensorData::new(noisy, [batch.len(), samples]), device);
        let clean = Tensor::<2>::from_data(TensorData::new(clean, [batch.len(), samples]), device);
        let prediction = processor.forward(model, noisy);
        for (index, (pair, &valid)) in batch.iter().zip(&lengths).enumerate() {
            let value = loss
                .forward(
                    prediction.clone().slice([index..index + 1, 0..valid]),
                    clean.clone().slice([index..index + 1, 0..valid]),
                    &[valid],
                )
                .try_into_scalar::<f32>()?;
            ensure!(value.is_finite(), "评估损失不是有限数，文件 {}", pair.id);
            weighted_loss += f64::from(value) * valid as f64;
            total_samples += valid;
        }
    }
    Ok(Evaluation {
        loss: weighted_loss / total_samples as f64,
        items: count,
        samples: total_samples,
    })
}

fn print_corpus(corpus: &Corpus) {
    println!(
        "配对文件：训练 {}，验证 {}，官方测试 {}",
        corpus.train.len(),
        corpus.validation.len(),
        corpus.test.len()
    );
    println!(
        "验证说话人：{}；官方测试集不参与选模",
        corpus.validation_speakers.join(", ")
    );
}

pub(super) fn run(args: Args, segment_samples: usize) -> Result<()> {
    let dataset = fs::canonicalize(&args.dataset)
        .with_context(|| format!("无法打开数据集目录 {}", args.dataset.display()))?;
    let corpus = data::discover(&dataset, &args.validation_speakers)?;
    let inference_device = device(&args)?;
    inference_device.seed(args.seed);
    let training_device = inference_device.clone().autodiff();
    let mut model = AnrLite48::new(&training_device).train();
    let mut optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(5.0)))
        .init();
    let training_processor = AudioProcessor::new(&training_device);
    let training_loss = EnhancementLoss::new(&training_device);
    let inference_processor = AudioProcessor::new(&inference_device);
    let inference_loss = EnhancementLoss::new(&inference_device);
    let output = create_output(args.output_dir.as_deref())?;
    write_json(
        &output.join("config.json"),
        &json!({
            "arguments": args,
            "resolved_dataset": dataset,
            "device": format!("{inference_device:?}"),
            "segment_samples": segment_samples,
            "sample_rate": SAMPLE_RATE,
            "loss": { "compression": 0.3, "power_epsilon": 1e-8, "magnitude_mse": 1.0, "complex_mse": 1.0, "waveform_l1": 0.1 },
            "optimizer": { "name": "AdamW", "weight_decay": 1e-4, "gradient_clipping_norm": 5.0 },
            "checkpoint_contents": "model parameters and normalization state; not optimizer or RNG resume state",
        }),
    )?;
    write_json(
        &output.join("splits.json"),
        &json!({
            "train": split_description(&corpus.train),
            "validation": split_description(&corpus.validation),
            "test": split_description(&corpus.test),
        }),
    )?;
    let mut metrics = BufWriter::new(File::create(output.join("metrics.jsonl"))?);
    println!("ANR-Lite-48：从随机初始化开始训练，未加载预训练权重");
    println!("数据集：{}", dataset.display());
    println!(
        "执行设备：{inference_device:?}；参数统计：{}",
        model.num_params()
    );
    println!(
        "训练片段：{segment_samples} samples；batch-size：{}",
        args.batch_size
    );
    print_corpus(&corpus);
    if let Some(limit) = args.max_validation_items {
        println!(
            "快速检查：每轮仅验证前 {} 个文件，不是完整验证集结果",
            limit.min(corpus.validation.len())
        );
    }
    println!("输出目录：{}", output.display());

    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut order: Vec<usize> = (0..corpus.train.len()).collect();
    let mut steps = 0usize;
    let mut summary = Summary {
        epoch: 0,
        epochs_completed: 0,
        steps: 0,
        partial_epoch: false,
        train_items: corpus.train.len(),
        validation_items: corpus.validation.len(),
        test_items: corpus.test.len(),
        validation_speakers: corpus.validation_speakers.clone(),
        validation_evaluated_items: 0,
        best_validation_loss: f64::INFINITY,
        best_epoch: 0,
        best_step: 0,
        best_checkpoint: "best.bin",
        last_checkpoint: "last.bin",
        test_loss: None,
        test_evaluated_items: 0,
    };
    for epoch in 1..=args.epochs {
        order.shuffle(&mut rng);
        let mut weighted_loss = 0.0;
        let mut epoch_samples = 0usize;
        let mut batches_done = 0usize;
        for indices in order.chunks(args.batch_size) {
            let batch = indices.len();
            let mut noisy = vec![0.0; batch * segment_samples];
            let mut clean = vec![0.0; batch * segment_samples];
            let mut valid_samples = Vec::with_capacity(batch);
            for (index, &pair_index) in indices.iter().enumerate() {
                let pair = &corpus.train[pair_index];
                let start = rng.random_range(0..=pair.samples.saturating_sub(segment_samples));
                let range = index * segment_samples..(index + 1) * segment_samples;
                valid_samples.push(read_segment(
                    pair,
                    start,
                    &mut noisy[range.clone()],
                    &mut clean[range],
                )?);
            }
            let samples: usize = valid_samples.iter().sum();
            let noisy = Tensor::<2>::from_data(
                TensorData::new(noisy, [batch, segment_samples]),
                &training_device,
            );
            let clean = Tensor::<2>::from_data(
                TensorData::new(clean, [batch, segment_samples]),
                &training_device,
            );
            let prediction = training_processor.forward(&model, noisy);
            let loss = training_loss.forward(prediction, clean, &valid_samples);
            let value = loss.clone().try_into_scalar::<f32>()?;
            ensure!(
                value.is_finite(),
                "训练损失不是有限数，epoch={epoch}，即将执行的 step={}",
                steps + 1
            );
            let gradients = GradientsParams::from_grads(loss.backward(), &model);
            ensure!(
                !gradients.is_empty(),
                "训练损失未产生模型参数梯度，未执行优化步骤"
            );
            model = optimizer.step(args.learning_rate, model, gradients);
            steps += 1;
            batches_done += 1;
            epoch_samples += samples;
            weighted_loss += f64::from(value) * samples as f64;
            serde_json::to_writer(
                &mut metrics,
                &json!({
                    "kind": "train", "epoch": epoch, "step": steps, "loss": value,
                    "items": batch, "valid_samples": samples,
                }),
            )?;
            writeln!(metrics)?;
            if steps == 1 || steps.is_multiple_of(args.log_every) {
                println!(
                    "epoch {epoch}/{}，step {steps}，train_loss={value:.6}",
                    args.epochs
                );
                metrics.flush()?;
            }
            if args.max_steps.is_some_and(|limit| steps >= limit) {
                break;
            }
        }
        let partial_epoch = batches_done < order.len().div_ceil(args.batch_size);
        let inference_model = model.clone().valid();
        let validation = evaluate(
            &inference_model,
            &corpus.validation,
            args.max_validation_items,
            &inference_processor,
            &inference_loss,
            &inference_device,
        )?;
        let improved = validation.loss < summary.best_validation_loss;
        inference_model
            .clone()
            .save_file(output.join("last.bin"))
            .context("保存 last.bin 失败")?;
        if improved {
            inference_model
                .save_file(output.join("best.bin"))
                .context("保存 best.bin 失败")?;
            summary.best_validation_loss = validation.loss;
            summary.best_epoch = epoch;
            summary.best_step = steps;
        }
        summary.epoch = epoch;
        summary.epochs_completed = if partial_epoch { epoch - 1 } else { epoch };
        summary.steps = steps;
        summary.partial_epoch = partial_epoch;
        summary.validation_evaluated_items = validation.items;
        let train_loss = weighted_loss / epoch_samples as f64;
        serde_json::to_writer(
            &mut metrics,
            &json!({
                "kind": "epoch", "epoch": epoch, "step": steps, "partial_epoch": partial_epoch,
                "train_loss": train_loss, "validation": validation, "best": improved,
            }),
        )?;
        writeln!(metrics)?;
        metrics.flush()?;
        write_json(&output.join("summary.json"), &summary)?;
        println!(
            "epoch {epoch}{}：train_loss={train_loss:.6}，val_loss={:.6}（{} 个文件）{}",
            if partial_epoch {
                "（未完成，达到步数上限）"
            } else {
                "完成"
            },
            validation.loss,
            validation.items,
            if improved {
                "，已保存新的 best.bin"
            } else {
                ""
            }
        );
        if args.max_steps.is_some_and(|limit| steps >= limit) {
            break;
        }
    }
    if args.evaluate_test {
        let best_model = AnrLite48::new(&inference_device)
            .try_load_file(output.join("best.bin"))
            .context("加载 best.bin 失败")?
            .valid();
        let evaluation = evaluate(
            &best_model,
            &corpus.test,
            args.max_test_items,
            &inference_processor,
            &inference_loss,
            &inference_device,
        )?;
        summary.test_loss = Some(evaluation.loss);
        summary.test_evaluated_items = evaluation.items;
        println!(
            "best.bin 测试损失：{:.6}（{}/{} 个官方测试文件{}）",
            evaluation.loss,
            evaluation.items,
            corpus.test.len(),
            if args.max_test_items.is_some() {
                "，已限制数量"
            } else {
                ""
            }
        );
        serde_json::to_writer(
            &mut metrics,
            &json!({ "kind": "test", "evaluation": evaluation }),
        )?;
        writeln!(metrics)?;
        metrics.flush()?;
        write_json(&output.join("summary.json"), &summary)?;
    }
    println!(
        "已执行 {steps} 次优化；模型保存在 {} 和 {}",
        output.join("best.bin").display(),
        output.join("last.bin").display()
    );
    println!("checkpoint 仅保存模型与归一化状态，不包含优化器或随机数状态，不能用于精确续训。");
    Ok(())
}

#[cfg(all(test, feature = "flex"))]
mod tests {
    use super::*;
    use anr_lite::burn::module::{ModuleMapper, Param};
    use hound::{SampleFormat, WavSpec, WavWriter};

    struct ZeroParameters;

    impl ModuleMapper for ZeroParameters {
        fn map_float<const D: usize>(&mut self, parameter: Param<Tensor<D>>) -> Param<Tensor<D>> {
            parameter.map(|tensor| Tensor::zeros(tensor.dims(), &tensor.device()))
        }
    }

    fn write_wave(path: &Path, samples: &[i16]) {
        let mut writer = WavWriter::create(
            path,
            WavSpec {
                channels: 1,
                sample_rate: SAMPLE_RATE,
                bits_per_sample: 16,
                sample_format: SampleFormat::Int,
            },
        )
        .unwrap();
        for &sample in samples {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn full_validation_preserves_unequal_length_weighting_and_unpadded_losses() {
        let temporary = tempfile::tempdir().unwrap();
        let device = Device::flex();
        let model = AnrLite48::new(&device).map(&mut ZeroParameters).valid();
        let processor = AudioProcessor::new(&device);
        let loss = EnhancementLoss::new(&device);
        let mut pairs = Vec::new();
        let mut expected_weighted = 0.0;
        let mut expected_samples = 0;
        let mut individual_losses = Vec::new();
        for (index, samples) in [517, 1073].into_iter().enumerate() {
            let noisy: Vec<i16> = (0..samples)
                .map(|sample| ((sample * 197 % 28001) as i32 - 14000) as i16)
                .collect();
            let clean: Vec<i16> = noisy
                .iter()
                .map(|&sample| if index == 0 { sample / 3 } else { -sample / 2 })
                .collect();
            let noisy_path = temporary.path().join(format!("noisy-{index}.wav"));
            let clean_path = temporary.path().join(format!("clean-{index}.wav"));
            write_wave(&noisy_path, &noisy);
            write_wave(&clean_path, &clean);
            // Every zero-logit ERB gain is 0.5 and every complex DF residual
            // is zero. The expected waveform does not use the audio processor.
            let prediction = Tensor::from_data(
                TensorData::new(
                    noisy
                        .iter()
                        .map(|&sample| f32::from(sample) / 65536.0)
                        .collect::<Vec<_>>(),
                    [1, samples],
                ),
                &device,
            );
            let target = Tensor::from_data(
                TensorData::new(
                    clean
                        .iter()
                        .map(|&sample| f32::from(sample) / 32768.0)
                        .collect::<Vec<_>>(),
                    [1, samples],
                ),
                &device,
            );
            let expected = f64::from(
                loss.forward(prediction, target, &[samples])
                    .try_into_scalar::<f32>()
                    .unwrap(),
            );
            individual_losses.push(expected);
            expected_weighted += expected * samples as f64;
            expected_samples += samples;
            pairs.push(Pair {
                id: format!("p226_{index}"),
                speaker: "p226".into(),
                noisy: noisy_path,
                clean: clean_path,
                samples,
            });
        }
        let full = evaluate(&model, &pairs, None, &processor, &loss, &device).unwrap();
        assert_eq!(full.items, pairs.len());
        assert_eq!(full.samples, expected_samples);
        assert!((full.loss - expected_weighted / expected_samples as f64).abs() < 1e-6);
        let limited = evaluate(&model, &pairs, Some(1), &processor, &loss, &device).unwrap();
        assert_eq!(limited.items, 1);
        assert_eq!(limited.samples, pairs[0].samples);
        assert!((limited.loss - individual_losses[0]).abs() < 1e-6);
    }
}
