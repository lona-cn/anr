use crate::{
    args::{Cli, Command, DenoiseArgs, InputFormat, OutputFormat, WavFormat},
    audio::{AudioSpec, input::open_input, output::create_output, pipeline},
    cancel,
    engine::{self, EnginePool},
};
use anyhow::{Context, Result, bail};
use file_id::{FileId, get_file_id};
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

struct DestinationGuard {
    protected_paths: HashSet<PathBuf>,
    protected_files: HashSet<FileId>,
    completed_files: HashSet<FileId>,
}

impl DestinationGuard {
    fn check(&self, output: &Path) -> Result<()> {
        if output == Path::new("-") {
            return Ok(());
        }
        self.check_resolved(output, &identity(output)?).map(|_| ())
    }

    fn check_resolved(&self, output: &Path, key: &Path) -> Result<Option<FileId>> {
        let file = existing_file_id(output)?;
        if self.protected_paths.contains(key)
            || file
                .as_ref()
                .is_some_and(|file| self.protected_files.contains(file))
        {
            return Err(usage(format!(
                "输出不能覆盖本次输入或模型：{}",
                output.display()
            )));
        }
        if file
            .as_ref()
            .is_some_and(|file| self.completed_files.contains(file))
        {
            bail!(
                "输出与本批次已完成的文件相同，拒绝再次覆盖：{}",
                output.display()
            );
        }
        Ok(file)
    }

    fn record(&mut self, output: &Path) -> Result<()> {
        if output != Path::new("-") {
            self.completed_files.insert(get_file_id(output)?);
        }
        Ok(())
    }
}

fn existing_file_id(path: &Path) -> Result<Option<FileId>> {
    match get_file_id(path) {
        Ok(file) => Ok(Some(file)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("无法识别文件身份：{}", path.display())),
    }
}

fn denoise(
    args: &DenoiseArgs,
    quiet: bool,
    verbose: bool,
    cancelled: Arc<AtomicBool>,
) -> Result<u8> {
    let default_model;
    let model = match args.model.as_deref() {
        Some(model) => model,
        None => {
            default_model =
                anr::default_model_path().context("未找到默认 DFN3 模型；可用 --model 指定文件")?;
            default_model.as_path()
        }
    };
    let (jobs, mut destinations) = plan_jobs(args, model)?;
    cancel::check(&cancelled)?;
    // Initialize once before creating any output directory or transaction.
    let mut engines = EnginePool::new(model, args.backend, args.provider, args.strength)?;
    if let Some(dir) = &args.output_dir {
        fs::create_dir_all(dir).with_context(|| format!("无法创建输出目录 {}", dir.display()))?;
        // Creating a previously missing parent can resolve `..` or aliases
        // differently. Recheck the same jobs before opening any output.
        destinations = validate_destinations(args, &jobs, model)?;
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
            destinations.check(&job.output)?;
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
                    "{}: {} Hz / {} 声道 → {} Hz，backend={}，strength={}",
                    job.input.display(),
                    input_spec.sample_rate,
                    input_spec.channels,
                    output_spec.sample_rate,
                    args.backend.name(),
                    args.strength
                );
                if args.backend == crate::args::Backend::Ort {
                    eprintln!(
                        "provider={}",
                        args.provider.unwrap_or(crate::args::Provider::Cpu).name()
                    );
                }
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
            destinations.check(&job.output)?;
            sink.finish()
                .with_context(|| format!("无法完成输出 {}", job.output.display()))?;
            if !quiet {
                eprintln!(
                    "已完成 {} → {}：每声道 {} 个采样，{} Hz，{} 声道，backend={}",
                    job.input.display(),
                    job.output.display(),
                    stats.output_frames,
                    stats.output_rate,
                    stats.channels,
                    args.backend.name()
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
        } else {
            destinations.record(&job.output).with_context(|| {
                format!(
                    "输出已保存，但无法核验其文件身份，已停止后续处理：{}",
                    job.output.display()
                )
            })?;
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

fn plan_jobs(args: &DenoiseArgs, model: &Path) -> Result<(Vec<Job>, DestinationGuard)> {
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
    let destinations = validate_destinations(args, &jobs, model)?;
    Ok((jobs, destinations))
}

fn validate_destinations(
    args: &DenoiseArgs,
    jobs: &[Job],
    model: &Path,
) -> Result<DestinationGuard> {
    let mut guard = DestinationGuard {
        protected_paths: HashSet::with_capacity(args.inputs.len() + 1),
        protected_files: HashSet::with_capacity(args.inputs.len() + 1),
        completed_files: HashSet::with_capacity(jobs.len()),
    };
    // Only audio INPUT "-" means stdin; --model "-" is a real file.
    for input in args
        .inputs
        .iter()
        .map(PathBuf::as_path)
        .filter(|input| *input != Path::new("-"))
        .chain(std::iter::once(model))
    {
        guard.protected_paths.insert(identity(input)?);
        if let Some(file) = existing_file_id(input)? {
            guard.protected_files.insert(file);
        }
    }
    let mut targets = HashSet::with_capacity(jobs.len());
    let mut target_files = HashSet::with_capacity(jobs.len());
    for job in jobs {
        let output = &job.output;
        if output == Path::new("-") {
            continue;
        }
        let key = identity(output)?;
        let file = guard.check_resolved(output, &key)?;
        if !targets.insert(key) || file.is_some_and(|file| !target_files.insert(file)) {
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
    Ok(guard)
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

pub fn is_cancelled(error: &anyhow::Error) -> bool {
    error.chain().any(|source| source.is::<cancel::Cancelled>())
}

pub fn is_broken_pipe(error: &anyhow::Error) -> bool {
    error.chain().any(|source| {
        if let Some(error) = source.downcast_ref::<io::Error>() {
            return error.kind() == io::ErrorKind::BrokenPipe;
        }
        source
            .downcast_ref::<serde_json::Error>()
            .is_some_and(|error| error.io_error_kind() == Some(io::ErrorKind::BrokenPipe))
    })
}
