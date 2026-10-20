use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result, ensure};
use rubato::{FftFixedInOut, Resampler};

use crate::cancel;

pub(super) fn rounded_frames(frames: u64, input_rate: u32, output_rate: u32) -> Result<u64> {
    ensure!(
        input_rate > 0 && output_rate > 0,
        "sample rates must be positive"
    );
    let count = (u128::from(frames) * u128::from(output_rate) + u128::from(input_rate) / 2)
        / u128::from(input_rate);
    u64::try_from(count).context("output frame count exceeds u64")
}

pub(super) fn ceiling_frames(frames: u64, input_rate: u32, output_rate: u32) -> Result<u64> {
    ensure!(
        input_rate > 0 && output_rate > 0,
        "sample rates must be positive"
    );
    let count = (u128::from(frames) * u128::from(output_rate)).div_ceil(u128::from(input_rate));
    u64::try_from(count).context("resampled frame count exceeds u64")
}

/// One bounded, exact-ratio FFT stream. The callback consumes its buffer before
/// returning; it must not retain samples. Same-rate streams never enter rubato.
pub(super) struct StreamResampler {
    fft: Option<FftFixedInOut<f32>>,
    input: Vec<Vec<f32>>,
    output: Vec<Vec<f32>>,
    channels: usize,
    input_rate: u32,
    output_rate: u32,
    filled: usize,
    delay_left: usize,
    input_frames: u64,
    emitted: u64,
}

impl StreamResampler {
    pub(super) fn new(input_rate: u32, output_rate: u32, channels: usize) -> Result<Self> {
        ensure!(
            input_rate > 0 && output_rate > 0,
            "sample rates must be positive"
        );
        ensure!(channels > 0, "audio must have at least one channel");
        let fft = if input_rate == output_rate {
            None
        } else {
            Some(
                FftFixedInOut::new(input_rate as usize, output_rate as usize, 1024, channels)
                    .context("constructing band-limited resampler")?,
            )
        };
        let (input, output, delay_left) = match &fft {
            Some(fft) => (
                fft.input_buffer_allocate(true),
                fft.output_buffer_allocate(true),
                fft.output_delay(),
            ),
            None => (Vec::new(), Vec::new(), 0),
        };
        Ok(Self {
            fft,
            input,
            output,
            channels,
            input_rate,
            output_rate,
            filled: 0,
            delay_left,
            input_frames: 0,
            emitted: 0,
        })
    }

    pub(super) fn push(
        &mut self,
        data: &[Vec<f32>],
        frames: usize,
        cancelled: &AtomicBool,
        emit: &mut impl FnMut(&[Vec<f32>], usize) -> Result<()>,
    ) -> Result<()> {
        cancel::check(cancelled)?;
        ensure!(
            data.len() == self.channels && data.iter().all(|ch| ch.len() >= frames),
            "invalid planar resampler input"
        );
        self.input_frames = self
            .input_frames
            .checked_add(frames as u64)
            .context("input frame count exceeds u64")?;
        if self.fft.is_none() {
            if frames > 0 {
                emit(data, frames)?;
                self.emitted = self
                    .emitted
                    .checked_add(frames as u64)
                    .context("output frame count exceeds u64")?;
            }
            return Ok(());
        }
        let chunk = self.input[0].len();
        let mut offset = 0;
        while offset < frames {
            cancel::check(cancelled)?;
            let take = (chunk - self.filled).min(frames - offset);
            for (buffer, channel) in self.input.iter_mut().zip(data) {
                buffer[self.filled..self.filled + take]
                    .copy_from_slice(&channel[offset..offset + take]);
            }
            self.filled += take;
            offset += take;
            if self.filled == chunk {
                self.process_chunk(None, emit)?;
                self.filled = 0;
            }
        }
        Ok(())
    }

    fn process_chunk(
        &mut self,
        target: Option<u64>,
        emit: &mut impl FnMut(&[Vec<f32>], usize) -> Result<()>,
    ) -> Result<()> {
        let fft = self
            .fft
            .as_mut()
            .expect("FFT chunk only on a resampling stream");
        let (consumed, produced) = fft
            .process_into_buffer(&self.input, &mut self.output, None)
            .context("resampling audio")?;
        ensure!(
            consumed == self.input[0].len(),
            "resampler did not consume its complete input block"
        );
        let skip = self.delay_left.min(produced);
        self.delay_left -= skip;
        let available = produced - skip;
        let frames = match target {
            Some(target) => {
                available.min(usize::try_from(target - self.emitted).unwrap_or(usize::MAX))
            }
            None => available,
        };
        if frames > 0 {
            if skip > 0 {
                for channel in &mut self.output {
                    channel.copy_within(skip..skip + frames, 0);
                }
            }
            emit(&self.output, frames)?;
            self.emitted = self
                .emitted
                .checked_add(frames as u64)
                .context("output frame count exceeds u64")?;
        }
        Ok(())
    }

    pub(super) fn finish(
        &mut self,
        target: u64,
        cancelled: &AtomicBool,
        emit: &mut impl FnMut(&[Vec<f32>], usize) -> Result<()>,
    ) -> Result<()> {
        cancel::check(cancelled)?;
        ensure!(
            target >= self.emitted,
            "resampling target precedes already emitted audio"
        );
        ensure!(
            target <= ceiling_frames(self.input_frames, self.input_rate, self.output_rate)?,
            "resampling target exceeds the input duration"
        );
        if self.fft.is_none() {
            ensure!(
                self.emitted == target,
                "same-rate stream has an inconsistent frame target"
            );
            return Ok(());
        }
        // rubato's process_partial_into_buffer allocates a temporary planar
        // input. Its documented equivalent is this reusable zero-padded block.
        for channel in &mut self.input {
            channel[self.filled..].fill(0.0);
        }
        while self.emitted < target {
            cancel::check(cancelled)?;
            self.process_chunk(Some(target), emit)?;
            self.filled = 0;
            for channel in &mut self.input {
                channel.fill(0.0);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn convert(input_rate: u32, output_rate: u32, input: &[f32], packet: usize) -> Vec<f32> {
        let mut converter = StreamResampler::new(input_rate, output_rate, 1).unwrap();
        let mut output = Vec::new();
        let mut block = vec![Vec::with_capacity(packet)];
        let cancelled = AtomicBool::new(false);
        let mut collect = |data: &[Vec<f32>], frames: usize| {
            output.extend_from_slice(&data[0][..frames]);
            Ok(())
        };
        for chunk in input.chunks(packet) {
            block[0].clear();
            block[0].extend_from_slice(chunk);
            converter
                .push(&block, chunk.len(), &cancelled, &mut collect)
                .unwrap();
        }
        let target = rounded_frames(input.len() as u64, input_rate, output_rate).unwrap();
        converter.finish(target, &cancelled, &mut collect).unwrap();
        output
    }

    #[test]
    fn delay_and_flush_preserve_a_tone_through_the_last_partial_chunk() {
        for (input_rate, output_rate, count, want) in [
            (16_000, 48_000, 16_003, 48_009),
            (44_100, 48_000, 44_107, 48_008),
            (48_000, 16_000, 48_007, 16_002),
            (48_000, 44_100, 48_007, 44_106),
        ] {
            let input: Vec<_> = (0..count)
                .map(|n| (std::f32::consts::TAU * 317. * n as f32 / input_rate as f32 + 0.7).sin())
                .collect();
            let output = convert(input_rate, output_rate, &input, 17);
            assert_eq!(output.len(), want);
            // The final 64 samples still contain signal, rather than zero-fill
            // standing in for unflushed filter history.
            let tail_energy = output[want - 64..].iter().map(|x| x * x).sum::<f32>() / 64.;
            assert!(
                tail_energy > 0.2,
                "missing tail for {input_rate}->{output_rate}: {tail_energy}"
            );
            let interior_error = output[128..want - 128]
                .iter()
                .enumerate()
                .map(|(i, value)| {
                    let expected = (std::f32::consts::TAU * 317. * (i + 128) as f32
                        / output_rate as f32
                        + 0.7)
                        .sin();
                    (value - expected).powi(2)
                })
                .sum::<f32>()
                / (want - 256) as f32;
            assert!(
                interior_error < 0.003,
                "uncompensated delay {input_rate}->{output_rate}: {interior_error}"
            );
        }
    }

    #[test]
    fn packet_splits_do_not_change_resampled_audio_or_accumulate_rounding() {
        let input: Vec<_> = (0..132_317).map(|n| (n as f32 * 0.013).sin()).collect();
        let small = convert(44_100, 16_000, &input, 7);
        let large = convert(44_100, 16_000, &input, 4096);
        assert_eq!(small.len(), 48_006);
        assert_eq!(small, large);
    }

    #[test]
    fn downsampling_rejects_energy_above_the_new_nyquist_frequency() {
        let input: Vec<_> = (0..48_000)
            .map(|n| (std::f32::consts::TAU * 12_000. * n as f32 / 48_000.).sin())
            .collect();
        let output = convert(48_000, 16_000, &input, 431);
        let energy = output[256..output.len() - 256]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            / (output.len() - 512) as f32;
        assert!(energy < 0.0001, "aliased out-of-band tone: {energy}");
    }

    #[test]
    fn equal_rate_bypass_is_exact_even_for_subnormal_samples() {
        let input = [f32::from_bits(1), -0.0, 0.25, -0.8, 0.12345679];
        let output = convert(48_000, 48_000, &input, 1);
        assert_eq!(
            output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            input.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn integer_targets_do_not_use_imprecise_float_duration() {
        assert_eq!(rounded_frames(1, 48_000, 16_000).unwrap(), 0);
        assert_eq!(rounded_frames(2, 48_000, 16_000).unwrap(), 1);
        assert_eq!(rounded_frames(44_107, 44_100, 48_000).unwrap(), 48_008);
        assert_eq!(rounded_frames(u64::MAX, 48_000, 48_000).unwrap(), u64::MAX);
        assert!(rounded_frames(u64::MAX, 1, 2).is_err());
    }
}
