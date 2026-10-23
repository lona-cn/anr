use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use serde::Serialize;

mod train;

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
enum Backend {
    Cpu,
    Wgpu,
}

#[derive(Debug, Parser, Serialize)]
#[command(
    name = "anr-lite-train",
    about = "使用 VoiceBank-DEMAND 28-speaker 训练 ANR-Lite-48"
)]
struct Args {
    /// 已解压的数据集根目录，包含四个标准 clean/noisy WAV 目录
    #[arg(long, default_value = "datasets/VoiceBank-DEMAND 28-speaker")]
    dataset: PathBuf,
    /// 新建的训练输出目录；不指定时在 target/anr-lite-training 下生成唯一目录
    #[arg(long)]
    output_dir: Option<PathBuf>,
    /// 明确选择执行设备；wgpu 需要编译时启用同名 feature，不会退回 CPU
    #[arg(long, value_enum, default_value = "cpu")]
    device: Backend,
    /// wgpu 独立 GPU 序号
    #[arg(long, default_value_t = 0)]
    gpu_index: usize,
    /// 最大训练轮数
    #[arg(long, default_value_t = 50)]
    epochs: usize,
    /// 每次优化使用的语音片段数
    #[arg(long, default_value_t = 160)]
    batch_size: usize,
    /// 随机训练片段时长；noisy 与 clean 使用相同裁剪位置
    #[arg(long, default_value_t = 4.0)]
    segment_seconds: f64,
    /// AdamW 学习率
    #[arg(long, default_value_t = 0.001)]
    learning_rate: f64,
    /// 模型初始化、训练顺序与裁剪的随机种子
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// 从训练目录留出的验证说话人；不使用官方测试集选模
    #[arg(long, value_delimiter = ',', num_args = 1.., default_value = "p226,p287")]
    validation_speakers: Vec<String>,
    /// 优化步数上限，用于短跑验收；到达上限后仍执行本轮验证和保存
    #[arg(long)]
    max_steps: Option<usize>,
    /// 限制验证文件数量，仅用于快速检查；默认验证全部留出文件
    #[arg(long)]
    max_validation_items: Option<usize>,
    /// 训练结束后加载 best.bin，在官方测试集上计算相同损失
    #[arg(long)]
    evaluate_test: bool,
    /// 限制最终测试文件数量，仅用于快速检查
    #[arg(long, requires = "evaluate_test")]
    max_test_items: Option<usize>,
    /// 每多少个优化步骤输出一次训练损失
    #[arg(long, default_value_t = 10)]
    log_every: usize,
}

impl Args {
    fn segment_samples(&self) -> Result<usize> {
        ensure!(self.epochs > 0, "epochs 必须大于零");
        ensure!(self.batch_size > 0, "batch-size 必须大于零");
        ensure!(self.log_every > 0, "log-every 必须大于零");
        ensure!(
            self.learning_rate.is_finite() && self.learning_rate > 0.0,
            "learning-rate 必须是大于零的有限数"
        );
        ensure!(
            self.segment_seconds.is_finite() && self.segment_seconds > 0.0,
            "segment-seconds 必须是大于零的有限数"
        );
        let samples = (self.segment_seconds * f64::from(anr_lite::SAMPLE_RATE)).round();
        ensure!(
            samples >= 1.0 && samples <= f64::from(u32::MAX),
            "segment-seconds 对应的采样数必须在 1..=u32::MAX 内"
        );
        let samples = samples as usize;
        samples
            .checked_mul(self.batch_size)
            .context("片段长度与 batch-size 的乘积溢出")?;
        for (name, value) in [
            ("max-steps", self.max_steps),
            ("max-validation-items", self.max_validation_items),
            ("max-test-items", self.max_test_items),
        ] {
            ensure!(value != Some(0), "{name} 必须大于零");
        }
        Ok(samples)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let segment_samples = args.segment_samples()?;
    train::run(args, segment_samples)
}
