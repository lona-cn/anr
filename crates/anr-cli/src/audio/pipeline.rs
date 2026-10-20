use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result, ensure};

use super::{
    AudioSink, AudioSource, ProcessStats,
    resample::{StreamResampler, ceiling_frames, rounded_frames},
};
use crate::{cancel, engine::EngineSpec};

/// Runs one source through independent channel states. File finalization belongs
/// to the command layer: no error or cancellation here can commit a sink.
pub fn process(
    source: &mut dyn AudioSource,
    sink: &mut dyn AudioSink,
    engines: &mut [Box<dyn anr::NoiseReduction>],
    engine_spec: EngineSpec,
    cancelled: &AtomicBool,
) -> Result<ProcessStats> {
    cancel::check(cancelled)?;
    let input_spec = source.info().spec;
    let output_spec = sink.spec();
    ensure!(
        input_spec.channels > 0,
        "audio must have at least one channel"
    );
    ensure!(
        input_spec.channels == output_spec.channels,
        "input and output channel counts differ"
    );
    ensure!(
        engines.len() == input_spec.channels,
        "one independent engine is required per channel"
    );
    ensure!(
        input_spec.sample_rate > 0 && output_spec.sample_rate > 0 && engine_spec.sample_rate > 0,
        "sample rates must be positive"
    );
    ensure!(
        engine_spec.frame_samples > 0,
        "engine frame size must be positive"
    );

    let mut to_model = StreamResampler::new(
        input_spec.sample_rate,
        engine_spec.sample_rate,
        input_spec.channels,
    )?;
    let mut model = ModelFrames::new(engines, engine_spec, input_spec.channels);
    let mut output = OutputStage::new(sink, engine_spec.sample_rate)?;
    let mut decoded: Vec<Vec<f32>> = (0..input_spec.channels)
        .map(|_| Vec::with_capacity(4096))
        .collect();
    let mut input_frames = 0_u64;
    for engine in model.engines.iter_mut() {
        engine.reset();
    }

    loop {
        cancel::check(cancelled)?;
        let frames = source
            .read_frames(&mut decoded)
            .context("reading input audio")?;
        cancel::check(cancelled)?;
        ensure!(
            source.info().spec == input_spec,
            "input audio format changed within the stream"
        );
        ensure!(
            decoded.len() == input_spec.channels && decoded.iter().all(|ch| ch.len() == frames),
            "source returned inconsistent planar frame counts"
        );
        if frames == 0 {
            break;
        }
        input_frames = input_frames
            .checked_add(frames as u64)
            .context("input frame count exceeds u64")?;
        output.writer.limit = rounded_frames(
            input_frames,
            input_spec.sample_rate,
            output_spec.sample_rate,
        )?;
        output.writer.flush_pending(cancelled)?;
        to_model.push(&decoded, frames, cancelled, &mut |data, count| {
            model.push(data, count, &mut output, cancelled)
        })?;
    }

    // Ceil in the intermediate model domain preserves the final fractional
    // sample's duration. Rounding twice could otherwise lose an output sample.
    let model_target = ceiling_frames(
        input_frames,
        input_spec.sample_rate,
        engine_spec.sample_rate,
    )?;
    to_model.finish(model_target, cancelled, &mut |data, count| {
        model.push(data, count, &mut output, cancelled)
    })?;
    model.finish(&mut output, cancelled)?;
    let output_frames = rounded_frames(
        input_frames,
        input_spec.sample_rate,
        output_spec.sample_rate,
    )?;
    output.finish(output_frames, cancelled)?;
    Ok(ProcessStats {
        input_frames,
        output_frames,
        input_rate: input_spec.sample_rate,
        output_rate: output_spec.sample_rate,
        channels: input_spec.channels,
    })
}

struct ModelFrames<'a> {
    engines: &'a mut [Box<dyn anr::NoiseReduction>],
    spec: EngineSpec,
    input: Vec<Vec<f32>>,
    output: Vec<Vec<f32>>,
    filled: usize,
}

impl<'a> ModelFrames<'a> {
    fn new(
        engines: &'a mut [Box<dyn anr::NoiseReduction>],
        spec: EngineSpec,
        channels: usize,
    ) -> Self {
        Self {
            engines,
            spec,
            input: vec![vec![0.0; spec.frame_samples]; channels],
            output: vec![vec![0.0; spec.frame_samples]; channels],
            filled: 0,
        }
    }

    fn push(
        &mut self,
        data: &[Vec<f32>],
        frames: usize,
        output: &mut OutputStage<'_>,
        cancelled: &AtomicBool,
    ) -> Result<()> {
        let mut offset = 0;
        while offset < frames {
            cancel::check(cancelled)?;
            let take = (self.spec.frame_samples - self.filled).min(frames - offset);
            for (buffer, channel) in self.input.iter_mut().zip(data) {
                buffer[self.filled..self.filled + take]
                    .copy_from_slice(&channel[offset..offset + take]);
            }
            self.filled += take;
            offset += take;
            if self.filled == self.spec.frame_samples {
                self.run_frame(self.spec.frame_samples, output, cancelled)?;
                self.filled = 0;
            }
        }
        Ok(())
    }

    fn run_frame(
        &mut self,
        valid: usize,
        output: &mut OutputStage<'_>,
        cancelled: &AtomicBool,
    ) -> Result<()> {
        for (channel, ((engine, input), result)) in self
            .engines
            .iter_mut()
            .zip(&self.input)
            .zip(&mut self.output)
            .enumerate()
        {
            cancel::check(cancelled)?;
            engine
                .run(self.spec.sample_rate, input, result)
                .with_context(|| format!("processing audio channel {}", channel + 1))?;
        }
        output.push(&self.output, valid, cancelled)
    }

    fn finish(&mut self, output: &mut OutputStage<'_>, cancelled: &AtomicBool) -> Result<()> {
        if self.filled > 0 {
            for channel in &mut self.input {
                channel[self.filled..].fill(0.0);
            }
            self.run_frame(self.filled, output, cancelled)?;
            self.filled = 0;
        }
        Ok(())
    }
}

struct OutputStage<'a> {
    resampler: StreamResampler,
    writer: LimitedWriter<'a>,
    model_rate: u32,
    output_rate: u32,
    model_frames: u64,
}

impl<'a> OutputStage<'a> {
    fn new(sink: &'a mut dyn AudioSink, model_rate: u32) -> Result<Self> {
        let spec = sink.spec();
        // At most one model sample is introduced by the intermediate ceil.
        // Keep its output-domain equivalent, never an unbounded audio queue.
        let pending_capacity =
            (u64::from(spec.sample_rate).div_ceil(u64::from(model_rate)) + 2) as usize;
        Ok(Self {
            resampler: StreamResampler::new(model_rate, spec.sample_rate, spec.channels)?,
            writer: LimitedWriter {
                sink,
                limit: 0,
                written: 0,
                pending: vec![vec![0.0; pending_capacity]; spec.channels],
                pending_frames: 0,
            },
            model_rate,
            output_rate: spec.sample_rate,
            model_frames: 0,
        })
    }

    fn push(&mut self, data: &[Vec<f32>], frames: usize, cancelled: &AtomicBool) -> Result<()> {
        self.model_frames = self
            .model_frames
            .checked_add(frames as u64)
            .context("model frame count exceeds u64")?;
        let writer = &mut self.writer;
        self.resampler
            .push(data, frames, cancelled, &mut |data, count| {
                writer.write(data, count, cancelled)
            })
    }

    fn finish(&mut self, target: u64, cancelled: &AtomicBool) -> Result<()> {
        self.writer.limit = target;
        // A bypass has no filter tail and forwards all intermediate ceil
        // samples; LimitedWriter is the final original-duration authority.
        let resampler_target = if self.model_rate == self.output_rate {
            self.model_frames
        } else {
            target
        };
        let writer = &mut self.writer;
        self.resampler
            .finish(resampler_target, cancelled, &mut |data, count| {
                writer.write(data, count, cancelled)
            })?;
        writer.flush_pending(cancelled)?;
        ensure!(
            writer.written == target,
            "audio pipeline ended before the exact output frame target"
        );
        Ok(())
    }
}

struct LimitedWriter<'a> {
    sink: &'a mut dyn AudioSink,
    limit: u64,
    written: u64,
    pending: Vec<Vec<f32>>,
    pending_frames: usize,
}

impl LimitedWriter<'_> {
    fn allowance(&self) -> usize {
        usize::try_from(self.limit - self.written).unwrap_or(usize::MAX)
    }

    fn flush_pending(&mut self, cancelled: &AtomicBool) -> Result<()> {
        let count = self.pending_frames.min(self.allowance());
        if count > 0 {
            cancel::check(cancelled)?;
            self.sink
                .write_frames(&self.pending, count)
                .context("writing output audio")?;
            self.written += count as u64;
            for channel in &mut self.pending {
                channel.copy_within(count..self.pending_frames, 0);
            }
            self.pending_frames -= count;
        }
        Ok(())
    }

    fn write(&mut self, data: &[Vec<f32>], frames: usize, cancelled: &AtomicBool) -> Result<()> {
        self.flush_pending(cancelled)?;
        let count = frames.min(self.allowance());
        if count > 0 {
            cancel::check(cancelled)?;
            self.sink
                .write_frames(data, count)
                .context("writing output audio")?;
            self.written += count as u64;
        }
        let extra = frames - count;
        ensure!(
            extra <= self.pending[0].len() - self.pending_frames,
            "resampled audio exceeded the bounded duration lookahead"
        );
        if extra > 0 {
            for (pending, channel) in self.pending.iter_mut().zip(data) {
                pending[self.pending_frames..self.pending_frames + extra]
                    .copy_from_slice(&channel[count..frames]);
            }
            self.pending_frames += extra;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{AudioInfo, AudioSpec};
    use std::io;

    struct Source {
        info: AudioInfo,
        data: Vec<Vec<f32>>,
        position: usize,
        chunks: Vec<usize>,
        next_chunk: usize,
    }

    impl Source {
        fn new(rate: u32, data: Vec<Vec<f32>>, chunks: Vec<usize>) -> Self {
            Self {
                info: AudioInfo {
                    format: "test".into(),
                    codec: "pcm".into(),
                    track_id: None,
                    spec: AudioSpec {
                        sample_rate: rate,
                        channels: data.len(),
                    },
                    frames: None,
                    duration_seconds: None,
                },
                data,
                position: 0,
                chunks,
                next_chunk: 0,
            }
        }
    }

    impl AudioSource for Source {
        fn info(&self) -> &AudioInfo {
            &self.info
        }
        fn read_frames(&mut self, output: &mut [Vec<f32>]) -> Result<usize> {
            let frames = self.chunks[self.next_chunk % self.chunks.len()]
                .min(self.data[0].len() - self.position);
            self.next_chunk += 1;
            for (out, input) in output.iter_mut().zip(&self.data) {
                out.clear();
                out.extend_from_slice(&input[self.position..self.position + frames]);
            }
            self.position += frames;
            Ok(frames)
        }
    }

    struct Sink {
        spec: AudioSpec,
        data: Vec<Vec<f32>>,
        broken: bool,
    }
    impl Sink {
        fn new(rate: u32, channels: usize) -> Self {
            Self {
                spec: AudioSpec {
                    sample_rate: rate,
                    channels,
                },
                data: vec![Vec::new(); channels],
                broken: false,
            }
        }
    }
    impl AudioSink for Sink {
        fn spec(&self) -> AudioSpec {
            self.spec
        }
        fn write_frames(&mut self, input: &[Vec<f32>], frames: usize) -> Result<()> {
            if self.broken {
                return Err(io::Error::from(io::ErrorKind::BrokenPipe).into());
            }
            for (out, input) in self.data.iter_mut().zip(input) {
                out.extend_from_slice(&input[..frames]);
            }
            Ok(())
        }
        fn finish(&mut self) -> Result<()> {
            panic!("pipeline must not commit the sink")
        }
    }

    // A frame-dependent state machine, not an echo: premature padding changes
    // every subsequent output, and failing to reset leaks the stale initial state.
    struct FrameState {
        frame: usize,
        accumulated: f32,
    }
    impl anr::NoiseReduction for FrameState {
        fn run(&mut self, rate: u32, input: &[f32], output: &mut [f32]) -> anr::Result<()> {
            if rate != 48_000 {
                return Err(anr::Error::InvalidSampleRate {
                    expected: 48_000,
                    actual: rate,
                });
            }
            if input.len() != self.frame {
                return Err(anr::Error::InvalidModel("incomplete engine frame".into()));
            }
            self.accumulated += input.iter().sum::<f32>();
            output.fill(self.accumulated);
            Ok(())
        }
        fn reset(&mut self) {
            self.accumulated = 0.0;
        }
        fn factor(&self) -> f32 {
            80.0
        }
        fn set_factor(&mut self, _: f32) {}
    }
    fn engines(channels: usize, frame: usize) -> Vec<Box<dyn anr::NoiseReduction>> {
        (0..channels)
            .map(|_| {
                Box::new(FrameState {
                    frame,
                    accumulated: 123.0,
                }) as Box<dyn anr::NoiseReduction>
            })
            .collect()
    }

    #[test]
    fn packet_boundaries_do_not_advance_model_frames_or_mix_channel_state() {
        let mut reducers = engines(2, 4);
        for chunks in [vec![1, 2, 1, 3], vec![9]] {
            let mut source = Source::new(
                48_000,
                vec![
                    vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
                    vec![2., 4., 6., 8., 10., 12., 14., 16., 18.],
                ],
                chunks,
            );
            let mut sink = Sink::new(48_000, 2);
            let stats = process(
                &mut source,
                &mut sink,
                &mut reducers,
                EngineSpec {
                    sample_rate: 48_000,
                    frame_samples: 4,
                },
                &AtomicBool::new(false),
            )
            .unwrap();
            assert_eq!(sink.data[0], [10., 10., 10., 10., 36., 36., 36., 36., 45.]);
            assert_eq!(sink.data[1], [20., 20., 20., 20., 72., 72., 72., 72., 90.]);
            assert_eq!((stats.input_frames, stats.output_frames), (9, 9));
        }
    }

    #[test]
    fn tiny_and_empty_streams_trim_padding_without_empty_model_runs() {
        for (data, expected) in [
            (vec![], vec![]),
            (vec![7.], vec![7.]),
            (vec![1., 2., 3.], vec![6., 6., 6.]),
        ] {
            let mut source = Source::new(48_000, vec![data], vec![1]);
            let mut sink = Sink::new(48_000, 1);
            process(
                &mut source,
                &mut sink,
                &mut engines(1, 4),
                EngineSpec {
                    sample_rate: 48_000,
                    frame_samples: 4,
                },
                &AtomicBool::new(false),
            )
            .unwrap();
            assert_eq!(sink.data[0], expected);
        }
    }

    #[test]
    fn all_rate_pairs_have_exact_total_counts_independent_of_packets() {
        for input_rate in [16_000, 44_100, 48_000] {
            for output_rate in [16_000, 44_100, 48_000] {
                for count in [1, 7, 479, 481, 96_013] {
                    let mut source =
                        Source::new(input_rate, vec![vec![0.125; count]], vec![1, 17, 503, 4096]);
                    let mut sink = Sink::new(output_rate, 1);
                    let stats = process(
                        &mut source,
                        &mut sink,
                        &mut engines(1, 480),
                        EngineSpec {
                            sample_rate: 48_000,
                            frame_samples: 480,
                        },
                        &AtomicBool::new(false),
                    )
                    .unwrap();
                    let want = ((count as u128 * output_rate as u128 + input_rate as u128 / 2)
                        / input_rate as u128) as usize;
                    assert_eq!(
                        sink.data[0].len(),
                        want,
                        "{input_rate}->{output_rate}, N={count}"
                    );
                    assert_eq!(stats.output_frames, want as u64);
                    assert!(sink.data[0].iter().all(|v| v.is_finite()));
                    if want > 0 {
                        assert!(
                            sink.data[0][want - 1].abs() > 0.00001,
                            "truncated tail {input_rate}->{output_rate}, N={count}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn sink_errors_and_cancellation_keep_their_concrete_causes() {
        let mut source = Source::new(48_000, vec![vec![1.; 4]], vec![4]);
        let mut sink = Sink::new(48_000, 1);
        sink.broken = true;
        let error = process(
            &mut source,
            &mut sink,
            &mut engines(1, 4),
            EngineSpec {
                sample_rate: 48_000,
                frame_samples: 4,
            },
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert_eq!(
            error.downcast_ref::<io::Error>().unwrap().kind(),
            io::ErrorKind::BrokenPipe
        );
        let error = process(
            &mut source,
            &mut sink,
            &mut engines(1, 4),
            EngineSpec {
                sample_rate: 48_000,
                frame_samples: 4,
            },
            &AtomicBool::new(true),
        )
        .unwrap_err();
        assert!(error.downcast_ref::<crate::cancel::Cancelled>().is_some());
    }

    #[test]
    fn invalid_channel_layout_or_engine_configuration_is_rejected_before_reading() {
        let spec = EngineSpec {
            sample_rate: 48_000,
            frame_samples: 4,
        };
        let mut source = Source::new(48_000, vec![vec![1.; 4]], vec![4]);
        let mut stereo_sink = Sink::new(48_000, 2);
        assert!(
            process(
                &mut source,
                &mut stereo_sink,
                &mut engines(1, 4),
                spec,
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert_eq!(source.position, 0);

        let mut sink = Sink::new(48_000, 1);
        assert!(
            process(
                &mut source,
                &mut sink,
                &mut [],
                spec,
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert!(
            process(
                &mut source,
                &mut sink,
                &mut engines(1, 4),
                EngineSpec {
                    sample_rate: 0,
                    frame_samples: 4
                },
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert!(
            process(
                &mut source,
                &mut sink,
                &mut engines(1, 4),
                EngineSpec {
                    sample_rate: 48_000,
                    frame_samples: 0
                },
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert_eq!(source.position, 0);
        assert!(sink.data[0].is_empty());
    }
}
