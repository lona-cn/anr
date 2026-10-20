use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
};

use anyhow::{Context, Result, bail, ensure};
use tempfile::NamedTempFile;

use super::{
    AudioSink, AudioSpec,
    stdio::{CancellableWriter, stdout_file},
};
use crate::{
    args::{OutputFormat, WavFormat},
    cancel,
};

const CHUNK_FRAMES: usize = 4096;

pub fn create_output(
    path: &Path,
    format: OutputFormat,
    wav_format: WavFormat,
    spec: AudioSpec,
    overwrite: bool,
    cancelled: Arc<AtomicBool>,
) -> Result<Box<dyn AudioSink>> {
    cancel::check(&cancelled)?;
    ensure!(
        spec.sample_rate > 0 && spec.channels > 0,
        "invalid output audio specification"
    );
    let frame_bytes = spec
        .channels
        .checked_mul(4)
        .context("output channel count is too large")?;
    let capacity = frame_bytes
        .checked_mul(CHUNK_FRAMES)
        .context("output buffer is too large")?;
    if path == Path::new("-") {
        ensure!(
            format == OutputFormat::F32le,
            "stdout supports only raw f32le"
        );
        return Ok(Box::new(StdoutSink {
            spec,
            writer: CancellableWriter::spawn(stdout_file()?, cancelled.clone())?,
            bytes: Vec::with_capacity(capacity),
            cancelled,
            finished: false,
            failed: false,
        }));
    }
    if !overwrite {
        ensure!(
            !path.try_exists()?,
            "output already exists: {}",
            path.display()
        );
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let temporary = NamedTempFile::new_in(parent)
        .with_context(|| format!("creating output temporary file in {}", parent.display()))?;
    let writer = BufWriter::new(temporary.as_file().try_clone()?);
    let encoder = match format {
        OutputFormat::F32le => Encoder::Raw(writer),
        OutputFormat::Wav => {
            let channels = u16::try_from(spec.channels).context("too many channels for WAV")?;
            let (bits_per_sample, sample_format) = match wav_format {
                WavFormat::Pcm16 => (16, hound::SampleFormat::Int),
                WavFormat::Pcm24 => (24, hound::SampleFormat::Int),
                WavFormat::Float32 => (32, hound::SampleFormat::Float),
            };
            let block_align = u32::from(channels) * u32::from(bits_per_sample / 8);
            ensure!(
                block_align <= u32::from(u16::MAX),
                "WAV block alignment exceeds RIFF limits"
            );
            spec.sample_rate
                .checked_mul(block_align)
                .context("WAV byte rate exceeds RIFF limits")?;
            Encoder::Wav(
                hound::WavWriter::new(
                    writer,
                    hound::WavSpec {
                        channels,
                        sample_rate: spec.sample_rate,
                        bits_per_sample,
                        sample_format,
                    },
                )?,
                wav_format,
            )
        }
    };
    Ok(Box::new(FileSink {
        spec,
        destination: path.to_owned(),
        temporary: Some(temporary),
        encoder: Some(encoder),
        bytes: Vec::with_capacity(capacity),
        overwrite,
        cancelled,
        finished: false,
        failed: false,
        wav_data_bytes: 0,
    }))
}

fn validate(input: &[Vec<f32>], frames: usize, spec: AudioSpec) -> Result<()> {
    ensure!(input.len() == spec.channels, "output channel count changed");
    for channel in input {
        ensure!(
            channel.len() >= frames,
            "output channel buffer is shorter than frame count"
        );
        ensure!(
            channel[..frames].iter().all(|sample| sample.is_finite()),
            "non-finite output audio sample"
        );
    }
    Ok(())
}

fn serialize(input: &[Vec<f32>], start: usize, end: usize, bytes: &mut Vec<u8>) {
    bytes.clear();
    for frame in start..end {
        for channel in input {
            bytes.extend_from_slice(&channel[frame].to_le_bytes());
        }
    }
}

enum Encoder {
    Raw(BufWriter<File>),
    Wav(hound::WavWriter<BufWriter<File>>, WavFormat),
}

struct FileSink {
    spec: AudioSpec,
    destination: PathBuf,
    // Close the encoder before removing its temporary path, including on Windows.
    encoder: Option<Encoder>,
    temporary: Option<NamedTempFile>,
    bytes: Vec<u8>,
    overwrite: bool,
    cancelled: Arc<AtomicBool>,
    finished: bool,
    failed: bool,
    wav_data_bytes: u64,
}

impl FileSink {
    fn write_inner(&mut self, input: &[Vec<f32>], frames: usize) -> Result<()> {
        ensure!(
            !self.finished && !self.failed,
            "output is no longer writable"
        );
        cancel::check(&self.cancelled)?;
        validate(input, frames, self.spec)?;
        let encoder = self.encoder.as_mut().context("output encoder is closed")?;
        if let Encoder::Wav(_, format) = encoder {
            let sample_bytes = match format {
                WavFormat::Pcm16 => 2_u64,
                WavFormat::Pcm24 => 3,
                WavFormat::Float32 => 4,
            };
            let added = u64::try_from(frames)?
                .checked_mul(self.spec.channels as u64)
                .and_then(|samples| samples.checked_mul(sample_bytes))
                .context("WAV data length overflow")?;
            let total = self
                .wav_data_bytes
                .checked_add(added)
                .context("WAV data length overflow")?;
            let header_bytes = if self.spec.channels > 2 || sample_bytes > 2 {
                60
            } else {
                36
            };
            ensure!(
                total <= u64::from(u32::MAX) - header_bytes,
                "WAV output exceeds RIFF 32-bit length limits"
            );
            self.wav_data_bytes = total;
        }
        for start in (0..frames).step_by(CHUNK_FRAMES) {
            cancel::check(&self.cancelled)?;
            let end = (start + CHUNK_FRAMES).min(frames);
            match encoder {
                Encoder::Raw(writer) => {
                    serialize(input, start, end, &mut self.bytes);
                    writer.write_all(&self.bytes)?;
                }
                Encoder::Wav(writer, format) => {
                    for frame in start..end {
                        for channel in input {
                            let value = channel[frame];
                            match format {
                                WavFormat::Float32 => writer.write_sample(value)?,
                                WavFormat::Pcm16 => writer.write_sample(
                                    (value.clamp(-1.0, 1.0) * 32768.0)
                                        .round()
                                        .clamp(-32768.0, 32767.0)
                                        as i16,
                                )?,
                                WavFormat::Pcm24 => writer.write_sample(
                                    (value.clamp(-1.0, 1.0) * 8388608.0)
                                        .round()
                                        .clamp(-8388608.0, 8388607.0)
                                        as i32,
                                )?,
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn finish_inner(&mut self) -> Result<()> {
        ensure!(!self.failed, "cannot commit failed output");
        if self.finished {
            return Ok(());
        }
        cancel::check(&self.cancelled)?;
        match self.encoder.take().context("output encoder is closed")? {
            Encoder::Raw(mut writer) => writer.flush()?,
            Encoder::Wav(writer, _) => writer.finalize()?,
        }
        cancel::check(&self.cancelled)?;
        let temporary = self
            .temporary
            .take()
            .context("output transaction is closed")?;
        let result = if self.overwrite {
            temporary.persist(&self.destination)
        } else {
            temporary.persist_noclobber(&self.destination)
        };
        // PersistError owns the still-temporary file. Dropping it cleans up on
        // failure; extract the original IO error without losing its kind.
        if let Err(error) = result {
            return Err(error.error)
                .with_context(|| format!("committing output {}", self.destination.display()));
        }
        self.finished = true;
        Ok(())
    }
}

impl AudioSink for FileSink {
    fn spec(&self) -> AudioSpec {
        self.spec
    }

    fn write_frames(&mut self, input: &[Vec<f32>], frames: usize) -> Result<()> {
        let result = self.write_inner(input, frames);
        self.failed |= result.is_err();
        result
    }

    fn finish(&mut self) -> Result<()> {
        let result = self.finish_inner();
        self.failed |= result.is_err();
        result
    }
}

struct StdoutSink {
    spec: AudioSpec,
    writer: CancellableWriter,
    bytes: Vec<u8>,
    cancelled: Arc<AtomicBool>,
    finished: bool,
    failed: bool,
}

impl AudioSink for StdoutSink {
    fn spec(&self) -> AudioSpec {
        self.spec
    }

    fn write_frames(&mut self, input: &[Vec<f32>], frames: usize) -> Result<()> {
        let result = (|| {
            ensure!(
                !self.finished && !self.failed,
                "stdout is no longer writable"
            );
            cancel::check(&self.cancelled)?;
            validate(input, frames, self.spec)?;
            for start in (0..frames).step_by(CHUNK_FRAMES) {
                cancel::check(&self.cancelled)?;
                serialize(
                    input,
                    start,
                    (start + CHUNK_FRAMES).min(frames),
                    &mut self.bytes,
                );
                self.writer
                    .write_buffer(&mut self.bytes)
                    .map_err(stdout_error)?;
            }
            Ok(())
        })();
        self.failed |= result.is_err();
        result
    }

    fn finish(&mut self) -> Result<()> {
        if self.failed {
            bail!("cannot finish failed stdout");
        }
        if !self.finished {
            cancel::check(&self.cancelled)?;
            self.writer.flush().map_err(stdout_error)?;
            self.finished = true;
        }
        Ok(())
    }
}

fn stdout_error(error: std::io::Error) -> anyhow::Error {
    if error
        .get_ref()
        .is_some_and(|inner| inner.is::<cancel::Cancelled>())
    {
        cancel::Cancelled.into()
    } else {
        error.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spec() -> AudioSpec {
        AudioSpec {
            sample_rate: 48000,
            channels: 2,
        }
    }
    fn flag() -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }
    #[test]
    fn abandoned_and_failed_outputs_leave_existing_target_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("old.wav");
        std::fs::write(&path, b"original").unwrap();
        {
            let mut sink = create_output(
                &path,
                OutputFormat::Wav,
                WavFormat::Pcm16,
                spec(),
                true,
                flag(),
            )
            .unwrap();
            sink.write_frames(&[vec![0.5], vec![-0.5]], 1).unwrap();
        }
        assert_eq!(std::fs::read(&path).unwrap(), b"original");
        {
            let mut sink = create_output(
                &path,
                OutputFormat::Wav,
                WavFormat::Float32,
                spec(),
                true,
                flag(),
            )
            .unwrap();
            assert!(sink.write_frames(&[vec![f32::NAN], vec![0.0]], 1).is_err());
        }
        assert_eq!(std::fs::read(&path).unwrap(), b"original");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }
    #[test]
    fn competing_target_is_not_replaced_without_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("race.raw");
        let mut sink = create_output(
            &path,
            OutputFormat::F32le,
            WavFormat::Pcm16,
            spec(),
            false,
            flag(),
        )
        .unwrap();
        sink.write_frames(&[vec![0.5], vec![-0.5]], 1).unwrap();
        std::fs::write(&path, b"winner").unwrap();
        assert!(sink.finish().is_err());
        drop(sink);
        assert_eq!(std::fs::read(&path).unwrap(), b"winner");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }
    #[test]
    fn raw_commits_interleaved_little_endian_only_on_finish() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.raw");
        let mut sink = create_output(
            &path,
            OutputFormat::F32le,
            WavFormat::Pcm16,
            spec(),
            false,
            flag(),
        )
        .unwrap();
        sink.write_frames(&[vec![0.25, 1.0], vec![-0.5, -1.0]], 2)
            .unwrap();
        assert!(!path.exists());
        sink.finish().unwrap();
        assert_eq!(
            std::fs::read(path).unwrap(),
            [0.25_f32, -0.5, 1.0, -1.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        );
    }
    #[test]
    fn all_wav_encodings_finalize_with_correct_channels_and_clamped_samples() {
        for format in [WavFormat::Pcm16, WavFormat::Pcm24, WavFormat::Float32] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("out.wav");
            let mut sink =
                create_output(&path, OutputFormat::Wav, format, spec(), false, flag()).unwrap();
            sink.write_frames(&[vec![2.0, -2.0], vec![0.25, -0.5]], 2)
                .unwrap();
            sink.finish().unwrap();
            let mut reader = hound::WavReader::open(path).unwrap();
            assert_eq!(reader.spec().channels, 2);
            assert_eq!(reader.duration(), 2);
            match format {
                WavFormat::Float32 => assert_eq!(
                    reader
                        .samples::<f32>()
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap(),
                    [2.0, 0.25, -2.0, -0.5]
                ),
                WavFormat::Pcm16 => assert_eq!(
                    reader
                        .samples::<i16>()
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap(),
                    [32767, 8192, -32768, -16384]
                ),
                WavFormat::Pcm24 => assert_eq!(
                    reader
                        .samples::<i32>()
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap(),
                    [8388607, 2097152, -8388608, -4194304]
                ),
            }
        }
    }
    #[test]
    fn cancellation_before_commit_preserves_old_target() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("old.raw");
        std::fs::write(&path, b"old").unwrap();
        let cancel = flag();
        let mut sink = create_output(
            &path,
            OutputFormat::F32le,
            WavFormat::Pcm16,
            spec(),
            true,
            cancel.clone(),
        )
        .unwrap();
        sink.write_frames(&[vec![0.0], vec![0.0]], 1).unwrap();
        cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(sink.finish().unwrap_err().is::<crate::cancel::Cancelled>());
        drop(sink);
        assert_eq!(std::fs::read(path).unwrap(), b"old");
    }
    #[test]
    fn failed_write_cannot_later_commit_even_if_finish_is_called() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("invalid.wav");
        let mut sink = create_output(
            &path,
            OutputFormat::Wav,
            WavFormat::Pcm16,
            spec(),
            false,
            flag(),
        )
        .unwrap();
        assert!(sink.write_frames(&[vec![0.0], vec![f32::NAN]], 1).is_err());
        assert!(sink.finish().is_err());
        assert!(!path.exists());
    }

    #[test]
    fn wav_header_rejects_overflowing_byte_rate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("invalid.wav");
        assert!(
            create_output(
                &path,
                OutputFormat::Wav,
                WavFormat::Float32,
                AudioSpec {
                    sample_rate: u32::MAX,
                    channels: 2
                },
                false,
                flag()
            )
            .is_err()
        );
        assert!(!path.exists());
    }

    #[test]
    fn stdout_error_retains_cancellation_marker_in_anyhow_chain() {
        struct CancelWriter;
        impl Write for CancelWriter {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other(crate::cancel::Cancelled))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let cancelled = flag();
        let mut sink = StdoutSink {
            spec: spec(),
            writer: CancellableWriter::spawn(CancelWriter, cancelled.clone()).unwrap(),
            bytes: Vec::new(),
            cancelled,
            finished: false,
            failed: false,
        };
        assert!(
            sink.write_frames(&[vec![0.0], vec![0.0]], 1)
                .unwrap_err()
                .is::<crate::cancel::Cancelled>()
        );
    }
}
