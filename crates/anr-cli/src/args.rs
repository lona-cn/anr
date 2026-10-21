use clap::{Args, Parser, Subcommand, ValueEnum, error::ErrorKind};
use std::{
    io::{self, IsTerminal},
    path::{Path, PathBuf},
};

#[derive(Debug, Parser)]
#[command(name = "anr", version, about = "文件、批处理和管道音频降噪")]
pub struct Cli {
    #[arg(
        short,
        long,
        global = true,
        conflicts_with = "verbose",
        help = "只显示错误"
    )]
    pub quiet: bool,
    #[arg(
        short,
        long,
        global = true,
        conflicts_with = "quiet",
        help = "显示处理规格和统计"
    )]
    pub verbose: bool,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// 对文件或标准输入中的音频降噪
    Denoise(DenoiseArgs),
    /// 查看文件的音频规格，不加载模型
    Inspect {
        input: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// 列出本程序编译的后端和执行 provider
    Backends {
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Args)]
pub struct DenoiseArgs {
    #[arg(required = true, num_args = 1.., value_name = "INPUT")]
    pub inputs: Vec<PathBuf>,
    #[arg(
        short,
        long,
        required_unless_present = "output_dir",
        conflicts_with = "output_dir",
        value_name = "PATH"
    )]
    pub output: Option<PathBuf>,
    #[arg(
        long,
        required_unless_present = "output",
        conflicts_with = "output",
        value_name = "DIR"
    )]
    pub output_dir: Option<PathBuf>,
    #[arg(long, value_name = "ONNX")]
    pub model: PathBuf,
    #[arg(long, value_enum, default_value_t = Backend::Ort)]
    pub backend: Backend,
    #[arg(long, value_enum, help = "仅适用于 ORT 后端；省略时 ORT 默认 cpu")]
    pub provider: Option<Provider>,
    #[arg(long, default_value_t = 80.0, value_parser = strength, allow_hyphen_values = true)]
    pub strength: f32,
    #[arg(long, value_enum, default_value_t = InputFormat::Auto)]
    pub input_format: InputFormat,
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub input_rate: Option<u32>,
    #[arg(long, value_parser = clap::value_parser!(u16).range(1..))]
    pub input_channels: Option<u16>,
    #[arg(long, value_enum)]
    pub output_format: Option<OutputFormat>,
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub output_rate: Option<u32>,
    #[arg(long, value_enum)]
    pub wav_format: Option<WavFormat>,
    #[arg(long, help = "允许替换输出文件，但不允许覆盖输入")]
    pub overwrite: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum InputFormat {
    Auto,
    F32le,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    Wav,
    F32le,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum WavFormat {
    Pcm16,
    Pcm24,
    Float32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Backend {
    Ort,
    Gl,
    Gles,
    Vk,
    Wgpu,
    Dx,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Self::Ort => "ort",
            Self::Gl => "gl",
            Self::Gles => "gles",
            Self::Vk => "vk",
            Self::Wgpu => "wgpu",
            Self::Dx => "dx",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Provider {
    Cpu,
    Cuda,
    Tensorrt,
    Directml,
    Coreml,
    Xnnpack,
}

impl Provider {
    pub const ALL: [Self; 6] = [
        Self::Cpu,
        Self::Cuda,
        Self::Tensorrt,
        Self::Directml,
        Self::Coreml,
        Self::Xnnpack,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Tensorrt => "tensorrt",
            Self::Directml => "directml",
            Self::Coreml => "coreml",
            Self::Xnnpack => "xnnpack",
        }
    }
    pub fn compiled(self) -> bool {
        cfg!(feature = "ort")
            && match self {
                Self::Cpu => true,
                Self::Cuda => cfg!(feature = "ort-cuda"),
                Self::Tensorrt => cfg!(feature = "ort-tensorrt"),
                Self::Directml => cfg!(all(feature = "ort-directml", target_os = "windows")),
                Self::Coreml => cfg!(all(feature = "ort-coreml", target_os = "macos")),
                Self::Xnnpack => cfg!(feature = "ort-xnnpack"),
            }
    }
}

impl DenoiseArgs {
    pub fn validate(&self) -> Result<(), clap::Error> {
        let invalid = |message| clap::Error::raw(ErrorKind::ArgumentConflict, message);
        if self.backend != Backend::Ort && self.provider.is_some() {
            return Err(invalid("--provider 只适用于 ORT 后端。"));
        }
        let stdin = self.inputs.iter().any(|p| p == Path::new("-"));
        if self.inputs.len() > 1 && self.output.is_some() {
            return Err(invalid("多个输入必须使用 --output-dir，不能共用一个输出。"));
        }
        if stdin && (self.inputs.len() != 1 || self.output_dir.is_some()) {
            return Err(invalid(
                "stdin 只能作为单个输入，并需用 --output 指定输出。",
            ));
        }
        if stdin && io::stdin().is_terminal() {
            return Err(invalid("stdin 是交互终端；请传入文件或连接二进制管道。"));
        }
        match self.input_format {
            InputFormat::F32le if self.input_rate.is_none() || self.input_channels.is_none() => {
                return Err(invalid("f32le 输入需要 --input-rate 和 --input-channels。"));
            }
            InputFormat::Auto if self.input_rate.is_some() || self.input_channels.is_some() => {
                return Err(invalid(
                    "--input-rate/--input-channels 只适用于 --input-format f32le，不能改写音频头。",
                ));
            }
            _ => {}
        }
        if self.output_dir.as_deref() == Some(Path::new("-")) {
            return Err(invalid("--output-dir 需要目录；stdout 请使用 --output -。"));
        }
        let format = if let Some(path) = &self.output {
            if path == Path::new("-") {
                if self.output_format != Some(OutputFormat::F32le) {
                    return Err(invalid(
                        "stdout 必须显式使用 --output-format f32le；WAV 输出需要可 seek 的文件。",
                    ));
                }
                if io::stdout().is_terminal() {
                    return Err(invalid(
                        "不能向交互终端写入二进制音频；请重定向 stdout 或指定输出文件。",
                    ));
                }
            }
            self.format_for(path)?
        } else {
            self.output_format.unwrap_or(OutputFormat::Wav)
        };
        if format == OutputFormat::F32le && self.wav_format.is_some() {
            return Err(invalid("--wav-format 不适用于 f32le 输出。"));
        }
        Ok(())
    }

    pub fn format_for(&self, path: &Path) -> Result<OutputFormat, clap::Error> {
        if let Some(format) = self.output_format {
            return Ok(format);
        }
        let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
        if extension.eq_ignore_ascii_case("wav") {
            Ok(OutputFormat::Wav)
        } else if extension.eq_ignore_ascii_case("f32le") {
            Ok(OutputFormat::F32le)
        } else {
            Err(clap::Error::raw(
                ErrorKind::ValueValidation,
                "无法从输出扩展名判断格式；请指定 --output-format wav 或 f32le。",
            ))
        }
    }
}

fn strength(text: &str) -> Result<f32, String> {
    let value: f32 = text
        .parse()
        .map_err(|_| "strength 必须是 0..=100 的有限数值".to_owned())?;
    if value.is_finite() && (0.0..=100.0).contains(&value) {
        Ok(value)
    } else {
        Err("strength 必须是 0..=100 的有限数值".to_owned())
    }
}
