use crate::{
    args::{Args, Backend},
    benchmark::Measurement,
};
use anyhow::Result;
use serde::Serialize;
use std::io::Write;

#[derive(Debug, Serialize)]
pub struct Report<'a> {
    pub config: &'a Args,
    pub sample_rate: u32,
    pub frame_size: usize,
    pub input: &'static str,
    pub input_seed: u32,
    pub os: &'static str,
    pub arch: &'static str,
    pub debug_assertions: bool,
    pub results: Vec<BackendResult>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum BackendResult {
    Ok {
        backend: Backend,
        #[serde(flatten)]
        measurement: Measurement,
    },
    Error {
        backend: Backend,
        error: String,
    },
}

impl Report<'_> {
    pub fn has_errors(&self) -> bool {
        self.results
            .iter()
            .any(|result| matches!(result, BackendResult::Error { .. }))
    }

    pub fn write(&self, mut output: impl Write) -> Result<()> {
        if self.config.json {
            serde_json::to_writer_pretty(&mut output, self)?;
            writeln!(output)?;
            return Ok(());
        }
        writeln!(
            output,
            "DFN3 | {} Hz mono | {} samples/frame | warmup={} | frames={} | strength={}",
            self.sample_rate,
            self.frame_size,
            self.config.warmup,
            self.config.frames,
            self.config.strength,
        )?;
        writeln!(
            output,
            "输入：固定种子的合成正弦波与噪声。延迟为同步 run 耗时，包含后端执行和主机/设备数据传输。"
        )?;
        writeln!(
            output,
            "初始化与预热不计入测量；FPS/RTF 使用正式循环总耗时，RTF < 1 表示快于实时。"
        )?;
        if self.config.backends.iter().any(|backend| {
            matches!(
                backend,
                Backend::OrtCuda | Backend::OrtTensorrt | Backend::OrtDirectml
            )
        }) {
            writeln!(
                output,
                "ORT 允许部分算子使用 CPU；其耗时包含 CPU 计算和设备间传输，不是纯 GPU kernel 耗时。"
            )?;
        }
        writeln!(
            output,
            "{:<13} {:>10} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10} {:>8} {:>7}",
            "backend",
            "init(ms)",
            "min(ms)",
            "mean(ms)",
            "max(ms)",
            "p50(ms)",
            "p95(ms)",
            "p99(ms)",
            "FPS",
            "RTF",
            ">10ms",
        )?;
        for result in &self.results {
            match result {
                BackendResult::Ok {
                    backend,
                    measurement,
                } => {
                    let stats = &measurement.statistics;
                    writeln!(
                        output,
                        "{:<13} {:>10.1} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>10.1} {:>8.4} {:>7}",
                        backend.name(),
                        measurement.initialization_ms,
                        stats.min_ms,
                        stats.mean_ms,
                        stats.max_ms,
                        stats.p50_ms,
                        stats.p95_ms,
                        stats.p99_ms,
                        stats.frames_per_second,
                        stats.rtf,
                        stats.over_budget_frames,
                    )?;
                }
                BackendResult::Error { backend, error } => {
                    writeln!(output, "{:<13} 失败：{error}", backend.name())?;
                }
            }
        }
        Ok(())
    }
}
