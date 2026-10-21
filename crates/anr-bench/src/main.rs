mod args;
mod benchmark;
#[cfg(feature = "ort")]
mod ort_backend;
mod report;
mod statistics;

use args::Args;
use clap::Parser;
use report::{BackendResult, Report};
use std::{io, process::ExitCode};

fn main() -> ExitCode {
    let args = Args::parse();
    if cfg!(debug_assertions) {
        eprintln!("警告：当前构建启用了 debug assertions；请使用 --release 获取性能结果。");
    }
    let input = benchmark::synthetic_input();
    let mut benchmark = benchmark::Benchmark::default();
    let mut results = Vec::with_capacity(args.backends.len());
    for &backend in &args.backends {
        eprintln!(
            "正在测试 {}：预热 {} 帧，正式测量 {} 帧……",
            backend.name(),
            args.warmup,
            args.frames,
        );
        results.push(match benchmark.run(backend, &args, &input) {
            Ok(measurement) => BackendResult::Ok {
                backend,
                measurement,
            },
            Err(error) => BackendResult::Error {
                backend,
                error: format!("{error:#}"),
            },
        });
    }
    let report = Report {
        config: &args,
        sample_rate: anr_dfn3::SAMPLE_RATE,
        frame_size: anr_dfn3::FRAME_SIZE,
        input: "synthetic",
        input_seed: benchmark::INPUT_SEED,
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        debug_assertions: cfg!(debug_assertions),
        results,
    };
    if let Err(error) = report.write(io::stdout().lock()) {
        eprintln!("无法输出 benchmark 结果：{error:#}");
        return ExitCode::FAILURE;
    }
    if report.has_errors() {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
