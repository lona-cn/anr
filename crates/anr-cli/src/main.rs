mod args;
mod audio;
mod cancel;
mod commands;
mod engine;

use args::{Cli, Command};
use clap::Parser;
use std::{
    process::ExitCode,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            let code = error.exit_code() as u8;
            let _ = error.print();
            return ExitCode::from(code);
        }
    };
    if let Command::Denoise(args) = &cli.command
        && let Err(error) = args.validate()
    {
        let _ = error.print();
        return ExitCode::from(2);
    }
    let cancelled = Arc::new(AtomicBool::new(false));
    if !matches!(cli.command, Command::Backends { .. }) {
        let flag = Arc::clone(&cancelled);
        if let Err(error) = ctrlc::set_handler(move || flag.store(true, Ordering::Relaxed)) {
            eprintln!("无法设置取消处理：{error}");
            return ExitCode::from(1);
        }
    }
    // Return normally after command scopes unwind: temporary output owners must
    // be dropped before the process exits, including on Ctrl+C or BrokenPipe.
    match commands::run(&cli, Arc::clone(&cancelled)) {
        Ok(code) => ExitCode::from(code),
        Err(error) if commands::is_cancelled(&error) || cancelled.load(Ordering::Relaxed) => {
            eprintln!("操作已取消；当前文件未提交输出。");
            ExitCode::from(130)
        }
        Err(error) if commands::is_broken_pipe(&error) => ExitCode::SUCCESS,
        Err(error) => {
            if let Some(argument_error) = error.downcast_ref::<clap::Error>() {
                let _ = argument_error.print();
                ExitCode::from(2)
            } else {
                eprintln!("错误：{error:#}");
                ExitCode::from(1)
            }
        }
    }
}
