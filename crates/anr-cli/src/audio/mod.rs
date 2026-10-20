pub mod input;
pub mod output;
pub mod pipeline;
pub mod resample;
mod stdio;

use anyhow::Result;
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct AudioSpec {
    pub sample_rate: u32,
    pub channels: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct AudioInfo {
    pub format: String,
    pub codec: String,
    pub track_id: Option<u32>,
    #[serde(flatten)]
    pub spec: AudioSpec,
    pub frames: Option<u64>,
    pub duration_seconds: Option<f64>,
}

pub trait AudioSource {
    fn info(&self) -> &AudioInfo;
    // Fill reusable planar buffers; return valid frames per channel, 0 at EOF.
    fn read_frames(&mut self, output: &mut [Vec<f32>]) -> Result<usize>;
}

pub trait AudioSink {
    fn spec(&self) -> AudioSpec;
    fn write_frames(&mut self, input: &[Vec<f32>], frames: usize) -> Result<()>;
    // Finalize and commit only after the complete pipeline succeeds.
    fn finish(&mut self) -> Result<()>;
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ProcessStats {
    pub input_frames: u64,
    pub output_frames: u64,
    pub input_rate: u32,
    pub output_rate: u32,
    pub channels: usize,
}
