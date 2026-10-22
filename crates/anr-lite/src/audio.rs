use burn::tensor::{Device, Tensor, TensorData};

use crate::{
    AUDIO_DELAY_SAMPLES, AnrLite48, FFT_SIZE, FRAME_SIZE, Fft960, LOOKAHEAD_FRAMES, ModelState,
    SpectralProcessor, SpectralState,
};

/// Analysis, synthesis and neural state for one batch of continuous streams.
///
/// Create a new value with [`AudioProcessor::initial_state`] to reset a stream.
/// This state is separate from the model's trainable parameters and checkpoints.
#[derive(Clone, Debug)]
pub struct AudioState {
    analysis: Tensor<2>,
    synthesis: Tensor<2>,
    model: ModelState,
    spectral: SpectralState,
}

impl AudioState {
    /// Truncate the backward graph without resetting any signal history.
    ///
    /// Use between truncated-backpropagation segments, not between ordinary
    /// frames when gradients through earlier frames are required.
    pub fn detach(self) -> Self {
        Self {
            analysis: self.analysis.detach(),
            synthesis: self.synthesis.detach(),
            model: self.model.detach(),
            spectral: self.spectral.detach(),
        }
    }
}

/// Differentiable 48 kHz analysis, neural enhancement and overlap-add synthesis.
///
/// All signal operations stay on the selected Burn device. This is a model
/// implementation, not an allocation-free audio-callback implementation or a
/// pretrained noise reducer. No sample-rate conversion is performed.
#[derive(Clone, Debug)]
pub struct AudioProcessor {
    fft: Fft960,
    spectral: SpectralProcessor,
    window: Tensor<2>,
    device: Device,
}

impl AudioProcessor {
    /// Construct device-resident transform, window and ERB constants.
    pub fn new(device: &Device) -> Self {
        // Vorbis analysis/synthesis window: complementary squared half-windows
        // sum to one at 50% overlap, including the stream's initial zero pad.
        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|n| {
                let phase = std::f64::consts::PI * (n as f64 + 0.5) / FFT_SIZE as f64;
                (std::f64::consts::FRAC_PI_2 * phase.sin().powi(2)).sin() as f32
            })
            .collect();
        Self {
            fft: Fft960::new(device),
            spectral: SpectralProcessor::new(device),
            window: Tensor::from_data(TensorData::new(window, [1, FFT_SIZE]), device),
            device: device.clone(),
        }
    }

    /// Start `batch` independent streams on this processor's device.
    ///
    /// # Panics
    /// Panics if `batch` is zero.
    pub fn initial_state(&self, batch: usize) -> AudioState {
        assert!(batch > 0, "audio batch must be nonzero");
        AudioState {
            analysis: Tensor::zeros([batch, FRAME_SIZE], &self.device),
            synthesis: Tensor::zeros([batch, FRAME_SIZE], &self.device),
            model: ModelState::new(batch, &self.device),
            spectral: self.spectral.initial_state(batch),
        }
    }

    /// Consume `[batch, 480]` samples and return `[batch, 480]` delayed samples.
    ///
    /// The output is delayed by [`AUDIO_DELAY_SAMPLES`] (three hops): two
    /// spectral lookahead frames and one analysis/synthesis overlap hop. This
    /// does not include collecting the current input block from an audio device.
    /// Neither stream state nor the backward graph is reset between calls.
    ///
    /// # Panics
    /// Panics on an empty batch, a non-480-sample frame, or incompatible state
    /// shape/device. Model parameters must use the same device.
    pub fn forward_frame(
        &self,
        model: &AnrLite48,
        input: Tensor<2>,
        state: &mut AudioState,
    ) -> Tensor<2> {
        let [batch, samples] = input.dims();
        assert!(batch > 0, "audio batch must be nonzero");
        assert_eq!(samples, FRAME_SIZE, "audio frame must contain 480 samples");
        assert_eq!(input.device(), self.device, "audio input device mismatch");
        assert_eq!(model.device(), self.device, "audio model device mismatch");
        assert_eq!(
            state.analysis.dims(),
            [batch, FRAME_SIZE],
            "audio state batch mismatch"
        );
        assert_eq!(
            state.synthesis.dims(),
            [batch, FRAME_SIZE],
            "audio overlap batch mismatch"
        );
        assert_eq!(
            state.analysis.device(),
            self.device,
            "audio state device mismatch"
        );
        assert_eq!(
            state.synthesis.device(),
            self.device,
            "audio overlap device mismatch"
        );

        let analysis =
            Tensor::cat(vec![state.analysis.clone(), input.clone()], 1) * self.window.clone();
        let spectrum = self.fft.forward(analysis);
        let (coarse, fine) = self
            .spectral
            .features(spectrum.clone(), &mut state.spectral);
        let (prediction, model_state) = model.forward_frame(coarse, fine, state.model.clone());
        let enhanced = self.spectral.filter(
            spectrum,
            prediction.gains,
            prediction.coefficients,
            &mut state.spectral,
        );
        let synthesis = self.fft.inverse(enhanced) * self.window.clone();
        let output = synthesis.clone().slice([0..batch, 0..FRAME_SIZE]) + state.synthesis.clone();
        state.analysis = input;
        state.synthesis = synthesis.slice([0..batch, FRAME_SIZE..FFT_SIZE]);
        state.model = model_state;
        output
    }

    /// Feed the three terminal zero hops needed to emit delayed stream samples.
    ///
    /// Returns `[batch, AUDIO_DELAY_SAMPLES]`. State advances through those zero
    /// frames; call only at end of stream, not between adjacent chunks. A partial
    /// final input hop must first be right-padded to 480 samples by the caller.
    pub fn flush(&self, model: &AnrLite48, state: &mut AudioState) -> Tensor<2> {
        let batch = state.analysis.dims()[0];
        let mut frames = Vec::with_capacity(LOOKAHEAD_FRAMES + 1);
        for _ in 0..=LOOKAHEAD_FRAMES {
            frames.push(self.forward_frame(
                model,
                Tensor::zeros([batch, FRAME_SIZE], &self.device),
                state,
            ));
        }
        Tensor::cat(frames, 1)
    }

    /// Process `[batch, samples]` and return exactly the original aligned length.
    ///
    /// This runs the same causal frame processing as [`Self::forward_frame`],
    /// right-pads a partial final hop, flushes the terminal state, and removes
    /// the known streaming delay. It is suitable for waveform losses: gradients
    /// remain connected through the network, temporal state and transforms.
    /// This method always starts a new stream; use frame processing for a
    /// recording split across multiple calls.
    ///
    /// # Panics
    /// Panics if either input dimension is zero or its device is incompatible.
    pub fn forward(&self, model: &AnrLite48, input: Tensor<2>) -> Tensor<2> {
        let [batch, samples] = input.dims();
        assert!(
            batch > 0 && samples > 0,
            "audio batch and sample count must be nonzero"
        );
        assert_eq!(input.device(), self.device, "audio input device mismatch");
        let mut state = self.initial_state(batch);
        let frame_count = samples.div_ceil(FRAME_SIZE);
        let mut outputs = Vec::with_capacity(frame_count + 1);
        for index in 0..frame_count {
            let start = index * FRAME_SIZE;
            let end = (start + FRAME_SIZE).min(samples);
            let mut frame = input.clone().slice([0..batch, start..end]);
            if end - start < FRAME_SIZE {
                frame = Tensor::cat(
                    vec![
                        frame,
                        Tensor::zeros([batch, FRAME_SIZE - (end - start)], &self.device),
                    ],
                    1,
                );
            }
            outputs.push(self.forward_frame(model, frame, &mut state));
        }
        outputs.push(self.flush(model, &mut state));
        Tensor::cat(outputs, 1)
            .slice([0..batch, AUDIO_DELAY_SAMPLES..AUDIO_DELAY_SAMPLES + samples])
    }
}

#[cfg(all(test, feature = "flex"))]
mod tests {
    use super::*;
    use burn::module::{Module, ModuleMapper, Param};

    struct ZeroParameters;

    impl ModuleMapper for ZeroParameters {
        fn map_float<const D: usize>(&mut self, parameter: Param<Tensor<D>>) -> Param<Tensor<D>> {
            parameter.map(|tensor| Tensor::zeros(tensor.dims(), &tensor.device()))
        }
    }

    #[test]
    fn public_streaming_path_has_known_gain_delay_and_partial_tail() {
        let device = Device::flex();
        // A real network with zero parameters has sigmoid gains of 0.5 and
        // zero residuals. It exercises the public consumer, not a copied DSP.
        let model = AnrLite48::new(&device).map(&mut ZeroParameters).valid();
        let processor = AudioProcessor::new(&device);
        let mut state = processor.initial_state(1);
        let source: Vec<f32> = (0..4 * FRAME_SIZE + 137)
            .map(|i| {
                0.2 * (std::f32::consts::TAU * 733.0 * i as f32 / 48_000.0).sin()
                    + if i == 317 { 0.4 } else { 0.0 }
            })
            .collect();
        let mut output = Vec::new();
        for chunk in source.chunks(FRAME_SIZE) {
            let mut values = vec![0.0; FRAME_SIZE];
            values[..chunk.len()].copy_from_slice(chunk);
            let current = Tensor::<2>::from_data(TensorData::new(values, [1, FRAME_SIZE]), &device);
            output.extend(
                processor
                    .forward_frame(&model, current, &mut state)
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap(),
            );
        }
        output.extend(
            processor
                .flush(&model, &mut state)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
        );
        assert!(output[..AUDIO_DELAY_SAMPLES].iter().all(|x| x.abs() < 3e-5));
        let aligned = &output[AUDIO_DELAY_SAMPLES..AUDIO_DELAY_SAMPLES + source.len()];
        for (i, (&input, &actual)) in source.iter().zip(aligned).enumerate() {
            let expected = 0.5 * input;
            assert!(
                (expected - actual).abs() < 3e-5,
                "sample {i}: expected={expected}, actual={actual}"
            );
        }
        assert!(
            output[AUDIO_DELAY_SAMPLES + source.len()..]
                .iter()
                .all(|x| x.abs() < 3e-5)
        );
    }
}
