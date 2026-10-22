use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anr_lite::{
    AnrLite48, AudioProcessor, SAMPLE_RATE,
    burn::{
        module::Module,
        tensor::{Device, Tensor, TensorData},
    },
};
use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
enum Backend {
    Cpu,
    Wgpu,
}

#[derive(Debug, Parser)]
#[command(
    name = "anr-lite-evaluate",
    about = "Render aligned 48 kHz float32 WAVs using a trained ANR-Lite record"
)]
struct Args {
    /// A real Burn model record, such as the trainer's best.bin.
    #[arg(long)]
    checkpoint: PathBuf,
    /// Directory of mono 48000 Hz PCM16 input WAVs, sorted by filename.
    #[arg(long)]
    input_dir: PathBuf,
    /// New output directory; existing directories are never reused.
    #[arg(long)]
    output_dir: PathBuf,
    /// Explicit backend selection; unavailable backends are errors, not fallbacks.
    #[arg(long, value_enum, default_value = "cpu")]
    device: Backend,
    /// Discrete GPU index, used only with --device wgpu.
    #[arg(long, default_value_t = 0)]
    gpu_index: usize,
    /// Number of variable-length utterances per zero-padded batch.
    #[arg(long, default_value_t = 8)]
    batch_size: usize,
    /// Positive smoke-only cap; omit to render every input WAV.
    #[arg(long)]
    limit: Option<usize>,
}

#[derive(Serialize)]
struct RenderedFile {
    filename: String,
    input_samples: usize,
    output_samples: usize,
    batch_index: usize,
}

#[derive(Serialize)]
struct RenderedBatch {
    batch_index: usize,
    items: usize,
    padded_samples: usize,
    render_seconds: f64,
}

#[derive(Serialize)]
struct Manifest {
    checkpoint: PathBuf,
    checkpoint_bytes: u64,
    input_dir: PathBuf,
    device: Backend,
    actual_device: String,
    gpu_index: usize,
    sample_rate: u32,
    channels: u16,
    output_format: &'static str,
    inference_mode: &'static str,
    source_count: usize,
    evaluated_count: usize,
    batch_size: usize,
    limit: Option<usize>,
    render_seconds: f64,
    render_timing_scope: &'static str,
    total_seconds: f64,
    files: Vec<RenderedFile>,
    batches: Vec<RenderedBatch>,
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
                anyhow::bail!(
                    "CPU evaluation requires the flex feature; no backend fallback is allowed"
                )
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
                    "wgpu evaluation requires --features train,wgpu; no CPU fallback is allowed"
                )
            }
        }
    }
}

fn input_paths(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(directory)
        .with_context(|| format!("cannot read input directory {}", directory.display()))?
    {
        let entry = entry.context("cannot read input directory entry")?;
        let path = entry.path();
        if path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
            && entry
                .metadata()
                .with_context(|| format!("cannot inspect {}", path.display()))?
                .is_file()
        {
            ensure!(
                path.file_name().and_then(|name| name.to_str()).is_some(),
                "WAV filename is not valid Unicode: {}",
                path.display()
            );
            paths.push(path);
        }
    }
    paths.sort();
    ensure!(
        !paths.is_empty(),
        "input directory contains no WAV files: {}",
        directory.display()
    );
    Ok(paths)
}

fn open_wav(path: &Path) -> Result<WavReader<BufReader<File>>> {
    let reader =
        WavReader::open(path).with_context(|| format!("cannot open WAV {}", path.display()))?;
    let spec = reader.spec();
    ensure!(
        spec.channels == 1
            && spec.sample_rate == SAMPLE_RATE
            && spec.sample_format == SampleFormat::Int
            && spec.bits_per_sample == 16,
        "{}: expected mono 48000 Hz PCM16 WAV, got {spec:?}",
        path.display()
    );
    ensure!(reader.duration() > 0, "{}: WAV is empty", path.display());
    Ok(reader)
}

fn read_batch(paths: &[PathBuf]) -> Result<(Vec<f32>, Vec<usize>, usize)> {
    let mut readers = paths
        .iter()
        .map(|path| open_wav(path))
        .collect::<Result<Vec<_>>>()?;
    let lengths: Vec<usize> = readers
        .iter()
        .map(|reader| reader.duration() as usize)
        .collect();
    let samples = *lengths
        .iter()
        .max()
        .context("cannot render an empty batch")?;
    let elements = samples
        .checked_mul(paths.len())
        .context("padded batch size overflow")?;
    let mut values = vec![0.0; elements];
    for ((reader, path), (output, &length)) in readers
        .iter_mut()
        .zip(paths)
        .zip(values.chunks_exact_mut(samples).zip(&lengths))
    {
        let mut decoded = 0;
        for sample in reader.samples::<i16>() {
            let sample =
                sample.with_context(|| format!("cannot decode PCM16 WAV {}", path.display()))?;
            ensure!(
                decoded < length,
                "{}: WAV contains more samples than declared",
                path.display()
            );
            output[decoded] = f32::from(sample) / 32768.0;
            decoded += 1;
        }
        ensure!(
            decoded == length,
            "{}: truncated WAV: expected {length} samples, decoded {decoded}",
            path.display()
        );
    }
    Ok((values, lengths, samples))
}

fn write_wav(path: &Path, samples: &[f32]) -> Result<()> {
    let mut writer = WavWriter::create(
        path,
        WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        },
    )
    .with_context(|| format!("cannot create WAV {}", path.display()))?;
    for &sample in samples {
        writer
            .write_sample(sample)
            .with_context(|| format!("cannot write WAV {}", path.display()))?;
    }
    writer
        .finalize()
        .with_context(|| format!("cannot finalize WAV {}", path.display()))
}

fn run(args: Args) -> Result<()> {
    let started = Instant::now();
    ensure!(args.batch_size > 0, "batch-size must be positive");
    ensure!(args.limit != Some(0), "limit must be positive");
    ensure!(
        !args
            .output_dir
            .try_exists()
            .context("cannot inspect output directory")?,
        "output directory already exists: {}",
        args.output_dir.display()
    );
    let paths = input_paths(&args.input_dir)?;
    let evaluated_count = args.limit.unwrap_or(paths.len()).min(paths.len());
    let checkpoint = fs::canonicalize(&args.checkpoint)
        .with_context(|| format!("cannot resolve checkpoint {}", args.checkpoint.display()))?;
    let checkpoint_bytes = fs::metadata(&checkpoint)
        .context("cannot inspect checkpoint")?
        .len();
    let execution_device = device(&args)?;
    let model = AnrLite48::new(&execution_device)
        .try_load_file(&checkpoint)
        .with_context(|| format!("cannot load checkpoint {}", checkpoint.display()))?
        .valid();
    let processor = AudioProcessor::new(&execution_device);
    if let Some(parent) = args
        .output_dir
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).context("cannot create output parent directory")?;
    }
    fs::create_dir(&args.output_dir).with_context(|| {
        format!(
            "cannot create fresh output directory {}",
            args.output_dir.display()
        )
    })?;
    let mut manifest = Manifest {
        checkpoint,
        checkpoint_bytes,
        input_dir: fs::canonicalize(&args.input_dir).context("cannot resolve input directory")?,
        device: args.device,
        actual_device: format!("{execution_device:?}"),
        gpu_index: args.gpu_index,
        sample_rate: SAMPLE_RATE,
        channels: 1,
        output_format: "float32 WAV; no clipping, normalization, or resampling",
        inference_mode: "valid; frozen batch normalization; fresh state per utterance",
        source_count: paths.len(),
        evaluated_count,
        batch_size: args.batch_size,
        limit: args.limit,
        render_seconds: 0.0,
        render_timing_scope: "sum of batch wall times: tensor upload, AudioProcessor.forward, and completed waveform download; excludes WAV reading/writing and checkpoint loading",
        total_seconds: 0.0,
        files: Vec::with_capacity(evaluated_count),
        batches: Vec::new(),
    };
    for (batch_index, batch) in paths[..evaluated_count].chunks(args.batch_size).enumerate() {
        let (values, lengths, padded_samples) = read_batch(batch)?;
        let rendering = Instant::now();
        let input = Tensor::<2>::from_data(
            TensorData::new(values, [batch.len(), padded_samples]),
            &execution_device,
        );
        // forward creates independent per-utterance states and performs the
        // required flush/delay compensation. No activations leave the device.
        let output = processor
            .forward(&model, input)
            .into_data()
            .try_into_vec::<f32>()
            .context("cannot download rendered waveform")?;
        let render_seconds = rendering.elapsed().as_secs_f64();
        ensure!(
            output.len() == batch.len() * padded_samples,
            "rendered batch shape is incorrect"
        );
        ensure!(
            output.iter().all(|sample| sample.is_finite()),
            "batch {batch_index} produced nonfinite audio; no outputs from this batch were written"
        );
        for (index, (path, &length)) in batch.iter().zip(&lengths).enumerate() {
            let filename = path
                .file_name()
                .and_then(|name| name.to_str())
                .context("invalid WAV filename")?;
            let start = index * padded_samples;
            write_wav(
                &args.output_dir.join(filename),
                &output[start..start + length],
            )?;
            manifest.files.push(RenderedFile {
                filename: filename.to_owned(),
                input_samples: length,
                output_samples: length,
                batch_index,
            });
        }
        manifest.render_seconds += render_seconds;
        manifest.batches.push(RenderedBatch {
            batch_index,
            items: batch.len(),
            padded_samples,
            render_seconds,
        });
    }
    manifest.total_seconds = started.elapsed().as_secs_f64();
    let path = args.output_dir.join("manifest.json");
    let mut writer = BufWriter::new(File::create(&path).context("cannot create manifest.json")?);
    serde_json::to_writer_pretty(&mut writer, &manifest)
        .context("cannot serialize evaluation manifest")?;
    writer.write_all(b"\n")?;
    writer.flush().context("cannot flush evaluation manifest")?;
    println!(
        "Rendered {} of {} source WAVs in {:.3}s rendering time; outputs: {}",
        manifest.evaluated_count,
        manifest.source_count,
        manifest.render_seconds,
        args.output_dir.display()
    );
    Ok(())
}

fn main() -> Result<()> {
    run(Args::parse())
}
