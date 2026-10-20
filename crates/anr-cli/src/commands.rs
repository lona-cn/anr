use crate::{
    args::{Cli, Command, DenoiseArgs, InputFormat, OutputFormat, WavFormat},
    audio::{AudioSpec, input::open_input, output::create_output, pipeline},
    cancel,
    engine::{self, EnginePool},
};
use anyhow::{Context, Result, bail};
use std::{
    collections::HashSet,
    ffi::OsString,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
};

pub fn run(cli: &Cli, cancelled: Arc<AtomicBool>) -> Result<u8> {
    match &cli.command {
        Command::Backends { json } => {
            let backends = engine::backends();
            let mut stdout = io::stdout().lock();
            if *json {
                serde_json::to_writer_pretty(
                    &mut stdout,
                    &serde_json::json!({"backends": backends}),
                )?;
                writeln!(stdout)?;
            } else {
                for backend in backends {
                    writeln!(
                        stdout,
                        "{}: {}",
                        backend.name,
                        if backend.compiled {
                            "已编译"
                        } else {
                            "未编译"
                        }
                    )?;
                    for provider in backend.providers {
                        writeln!(
                            stdout,
                            "  {}: {}",
                            provider.name,
                            if provider.compiled {
                                "已编译，未检查运行时"
                            } else {
                                "未编译或不支持当前平台"
                            }
                        )?;
                    }
                }
            }
            Ok(0)
        }
        Command::Inspect { input, json } => {
            let source = open_input(input, InputFormat::Auto, None, cancelled)
                .with_context(|| format!("无法读取音频 {}", input.display()))?;
            let info = source.info();
            let mut stdout = io::stdout().lock();
            if *json {
                serde_json::to_writer_pretty(&mut stdout, info)?;
                writeln!(stdout)?;
            } else {
                writeln!(stdout, "文件: {}", input.display())?;
                writeln!(stdout, "格式: {} / {}", info.format, info.codec)?;
                writeln!(
                    stdout,
                    "采样率: {} Hz\n声道: {}",
                    info.spec.sample_rate, info.spec.channels
                )?;
                if let Some(id) = info.track_id {
                    writeln!(stdout, "音轨: {id}")?;
                }
                if let Some(duration) = info.duration_seconds {
                    writeln!(stdout, "头部声明时长: {duration:.6} 秒")?;
                } else {
                    writeln!(stdout, "时长: 未知")?;
                }
            }
            Ok(0)
        }
        Command::Denoise(args) => denoise(args, cli.quiet, cli.verbose, cancelled),
    }
}

struct Job {
    input: PathBuf,
    output: PathBuf,
    format: OutputFormat,
}

fn denoise(
    args: &DenoiseArgs,
    quiet: bool,
    verbose: bool,
    cancelled: Arc<AtomicBool>,
) -> Result<u8> {
    let jobs = plan_jobs(args)?;
    cancel::check(&cancelled)?;
    // Initialize once before creating any output directory or transaction.
    let mut engines = EnginePool::new(&args.model, args.provider, args.strength)?;
    if let Some(dir) = &args.output_dir {
        fs::create_dir_all(dir).with_context(|| format!("无法创建输出目录 {}", dir.display()))?;
        // Creating a previously missing parent can resolve `..` or aliases
        // differently. Recheck the same jobs before opening any output.
        validate_destinations(args, &jobs)?;
    }
    let raw_spec = args
        .input_rate
        .zip(args.input_channels)
        .map(|(sample_rate, channels)| AudioSpec {
            sample_rate,
            channels: usize::from(channels),
        });
    let mut failed = 0;
    for job in &jobs {
        cancel::check(&cancelled)?;
        let result = (|| -> Result<()> {
            let mut source = open_input(
                &job.input,
                args.input_format,
                raw_spec,
                Arc::clone(&cancelled),
            )
            .with_context(|| format!("无法读取输入 {}", job.input.display()))?;
            let input_spec = source.info().spec;
            engines
                .prepare(input_spec.channels)
                .context(EngineInitialization)?;
            let output_spec = AudioSpec {
                sample_rate: args.output_rate.unwrap_or(input_spec.sample_rate),
                channels: input_spec.channels,
            };
            if verbose {
                eprintln!(
                    "{}: {} Hz / {} 声道 → {} Hz，provider={}，strength={}",
                    job.input.display(),
                    input_spec.sample_rate,
                    input_spec.channels,
                    output_spec.sample_rate,
                    args.provider.name(),
                    args.strength
                );
            }
            let mut sink = create_output(
                &job.output,
                job.format,
                args.wav_format.unwrap_or(WavFormat::Pcm16),
                output_spec,
                args.overwrite,
                Arc::clone(&cancelled),
            )
            .with_context(|| format!("无法准备输出 {}", job.output.display()))?;
            let spec = engines.spec();
            let stats = pipeline::process(
                &mut *source,
                &mut *sink,
                engines.reducers(),
                spec,
                &cancelled,
            )?;
            cancel::check(&cancelled)?;
            sink.finish()
                .with_context(|| format!("无法完成输出 {}", job.output.display()))?;
            if !quiet {
                eprintln!(
                    "已完成 {} → {}：每声道 {} 个采样，{} Hz，{} 声道",
                    job.input.display(),
                    job.output.display(),
                    stats.output_frames,
                    stats.output_rate,
                    stats.channels
                );
            }
            Ok(())
        })();
        if let Err(error) = result {
            if is_cancelled(&error) || is_broken_pipe(&error) || error.is::<EngineInitialization>()
            {
                return Err(error);
            }
            failed += 1;
            eprintln!("{}：{error:#}", job.input.display());
        }
    }
    if failed > 0 {
        eprintln!(
            "{} 个文件失败，{} 个文件完成；失败文件未提交输出。",
            failed,
            jobs.len() - failed
        );
        Ok(1)
    } else {
        Ok(0)
    }
}

#[derive(Debug)]
struct EngineInitialization;

impl std::fmt::Display for EngineInitialization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("无法初始化独立声道的降噪引擎")
    }
}

fn usage(message: String) -> anyhow::Error {
    clap::Error::raw(clap::error::ErrorKind::ArgumentConflict, message).into()
}

fn plan_jobs(args: &DenoiseArgs) -> Result<Vec<Job>> {
    let mut jobs = Vec::with_capacity(args.inputs.len());
    for input in &args.inputs {
        let (output, format) = if let Some(output) = &args.output {
            (output.clone(), args.format_for(output)?)
        } else {
            let directory = args.output_dir.as_ref().context("缺少输出目录")?;
            let format = args.output_format.unwrap_or(OutputFormat::Wav);
            let mut name: OsString = input
                .file_stem()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| usage(format!("输入路径没有可用的文件名：{}", input.display())))?
                .to_owned();
            name.push(match format {
                OutputFormat::Wav => ".wav",
                OutputFormat::F32le => ".f32le",
            });
            (directory.join(name), format)
        };
        jobs.push(Job {
            input: input.clone(),
            output,
            format,
        });
    }
    validate_destinations(args, &jobs)?;
    Ok(jobs)
}

fn validate_destinations(args: &DenoiseArgs, jobs: &[Job]) -> Result<()> {
    let mut protected = HashSet::with_capacity(args.inputs.len() + 1);
    for input in &args.inputs {
        if input != Path::new("-") {
            protected.insert(target_key(&identity(input)?));
        }
    }
    // Only audio INPUT "-" means stdin; --model "-" is a real file.
    protected.insert(target_key(&identity(&args.model)?));
    let mut targets = HashSet::with_capacity(jobs.len());
    for job in jobs {
        let output = &job.output;
        if output == Path::new("-") {
            continue;
        }
        let key = target_key(&identity(output)?);
        if protected.contains(&key) {
            return Err(usage(format!(
                "输出不能覆盖本次输入或模型：{}",
                output.display()
            )));
        }
        if !targets.insert(key) {
            return Err(usage(format!(
                "多个输入对应同一个输出：{}；请分别指定输出位置",
                output.display()
            )));
        }
        match fs::symlink_metadata(output) {
            Ok(metadata) => {
                if metadata.is_dir() {
                    return Err(usage(format!("输出路径是目录：{}", output.display())));
                }
                if !args.overwrite {
                    bail!(
                        "输出已存在：{}；确认替换后再使用 --overwrite",
                        output.display()
                    );
                }
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| format!("无法检查输出 {}", output.display()));
            }
        }
    }
    Ok(())
}

fn identity(path: &Path) -> Result<PathBuf> {
    if let Ok(canonical) = fs::canonicalize(path) {
        return Ok(canonical);
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if let (Ok(parent), Some(name)) = (fs::canonicalize(parent), path.file_name()) {
        return Ok(parent.join(name));
    }
    std::path::absolute(path).with_context(|| format!("无法解析路径 {}", path.display()))
}

fn target_key(path: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        PathBuf::from(path.to_string_lossy().to_lowercase())
    }
    #[cfg(not(windows))]
    {
        path.to_owned()
    }
}

pub fn is_cancelled(error: &anyhow::Error) -> bool {
    error.chain().any(|source| source.is::<cancel::Cancelled>())
}

pub fn is_broken_pipe(error: &anyhow::Error) -> bool {
    error.chain().any(|source| {
        source
            .downcast_ref::<io::Error>()
            .is_some_and(|error| error.kind() == io::ErrorKind::BrokenPipe)
    })
}
