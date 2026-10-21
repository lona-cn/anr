use clap::{Parser, ValueEnum};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Debug, Parser, Serialize)]
#[command(
    name = "anr-bench",
    version,
    about = "比较原生 Compute 与 ORT GPU provider 的连续帧推理性能"
)]
pub struct Args {
    #[arg(
        long,
        value_name = "ONNX",
        default_value = "models/dfn3.onnx",
        help = "固定 DFN3 模型路径"
    )]
    pub model: PathBuf,
    #[arg(
        long = "backend",
        value_enum,
        value_delimiter = ',',
        default_value = "gl,vk,dx",
        help = "按给定顺序串行测试，可用逗号分隔或重复指定"
    )]
    pub backends: Vec<Backend>,
    #[arg(
        long,
        default_value_t = 100,
        help = "每个后端的预热帧数，不计入统计；允许为 0"
    )]
    pub warmup: u32,
    #[arg(long, default_value_t = 1_000, value_parser = clap::value_parser!(u32).range(1..), help = "每个后端正式测量的帧数")]
    pub frames: u32,
    #[arg(long, default_value_t = 80.0, value_parser = strength, allow_hyphen_values = true, help = "降噪强度，0–100")]
    pub strength: f32,
    #[arg(long, help = "向 stdout 输出 JSON；进度和警告写入 stderr")]
    #[serde(skip)]
    pub json: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Backend {
    Gl,
    Vk,
    Dx,
    OrtCuda,
    OrtTensorrt,
    OrtDirectml,
}

impl Backend {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Gl => "gl",
            Self::Vk => "vk",
            Self::Dx => "dx",
            Self::OrtCuda => "ort-cuda",
            Self::OrtTensorrt => "ort-tensorrt",
            Self::OrtDirectml => "ort-directml",
        }
    }
}

fn strength(value: &str) -> Result<f32, String> {
    let value: f32 = value
        .parse()
        .map_err(|_| "强度必须是 0–100 的有限数值".to_owned())?;
    if value.is_finite() && (0.0..=100.0).contains(&value) {
        Ok(value)
    } else {
        Err("强度必须是 0–100 的有限数值".to_owned())
    }
}
