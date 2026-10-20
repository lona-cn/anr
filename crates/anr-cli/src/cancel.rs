use std::{
    fmt,
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Debug)]
pub struct Cancelled;

impl fmt::Display for Cancelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("操作已取消")
    }
}
impl std::error::Error for Cancelled {}

pub fn check(cancelled: &AtomicBool) -> anyhow::Result<()> {
    if cancelled.load(Ordering::Relaxed) {
        return Err(Cancelled.into());
    }
    Ok(())
}
