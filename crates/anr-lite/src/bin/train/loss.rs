use anr_lite::{FFT_SIZE, FRAME_SIZE, FREQ_BINS, Fft960};
use burn::tensor::{Device, Int, Tensor, TensorData, ops::PadMode};

const COMPRESSION: f32 = 0.3;
const EPSILON: f32 = 1e-8;

/// Phase-sensitive compressed spectral loss plus a sample-normalized waveform loss.
///
/// With `z = FFT(Hann * frame) / sqrt(960)` and `p = re² + im² + 1e-8`,
/// the spectral terms compare `p^0.15` and `z * p^-0.35`. Each frame's
/// frequency/component-mean squared errors are weighted by its valid Hann
/// energy, `sum(valid * Hann²) / sum(Hann²)`, then divided by the sum of
/// those weights. Fully padded frames therefore contribute nothing. The
/// waveform term is `0.1 * sum(valid * abs(prediction - target)) / sum(valid)`.
pub(super) struct EnhancementLoss {
    fft: Fft960,
    window: Tensor<3>,
    window_energy: Tensor<3>,
    device: Device,
}

impl EnhancementLoss {
    pub fn new(device: &Device) -> Self {
        let device = device.clone().without_autodiff();
        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|n| {
                let phase = std::f64::consts::TAU * n as f64 / FFT_SIZE as f64;
                (0.5 - 0.5 * phase.cos()) as f32
            })
            .collect();
        let window: Tensor<3> =
            Tensor::from_data(TensorData::new(window, [1, 1, FFT_SIZE]), &device);
        Self {
            fft: Fft960::new(&device),
            window_energy: window.clone().square(),
            window,
            device,
        }
    }

    /// Compare aligned `[batch, samples]` waveforms with real, nonempty prefixes.
    pub fn forward(
        &self,
        prediction: Tensor<2>,
        target: Tensor<2>,
        valid_samples: &[usize],
    ) -> Tensor<1> {
        let [batch, samples] = prediction.dims();
        assert!(batch > 0 && samples > 0, "loss expects nonempty waveforms");
        assert_eq!(
            prediction.dims(),
            target.dims(),
            "waveform shapes must match"
        );
        assert_eq!(
            prediction.device(),
            self.device,
            "prediction device must match loss"
        );
        assert_eq!(
            target.device(),
            self.device,
            "target device must match loss"
        );
        assert_eq!(
            valid_samples.len(),
            batch,
            "one valid length is required per item"
        );
        assert!(
            valid_samples
                .iter()
                .all(|&length| length > 0 && length <= samples),
            "valid lengths must be in 1..=samples"
        );

        // Only B lengths cross from the host. Construct the B*N mask on-device,
        // outside autodiff, rather than uploading a host mask for each batch.
        let lengths: Vec<i64> = valid_samples.iter().map(|&length| length as i64).collect();
        let lengths =
            Tensor::<2, Int>::from_data(TensorData::new(lengths, [batch, 1]), &self.device);
        let positions =
            Tensor::<1, Int>::arange(0..samples as i64, &self.device).reshape([1, samples]);
        let invalid = positions.greater_equal(lengths);
        let valid = invalid.clone().bool_not().float();

        // Fill instead of multiplying by zero: even nonfinite padding is
        // removed before FFT and waveform arithmetic.
        let prediction = prediction.mask_fill(invalid.clone(), 0.0);
        let target = target.mask_fill(invalid, 0.0);
        let waveform = (prediction.clone() - target.clone()).abs().sum()
            / valid_samples.iter().sum::<usize>() as f32;

        // Center the first frame on sample zero. Complete the final hop before
        // adding the right half-window so every real sample has both boundary
        // frames, regardless of the fixed batch's padded sample capacity.
        let hops = samples.div_ceil(FRAME_SIZE);
        let right_padding = hops * FRAME_SIZE - samples + FRAME_SIZE;
        let frame_weights = (valid
            .pad([(FRAME_SIZE, right_padding)], PadMode::Constant(0.0))
            .unfold::<3, _>(1, FFT_SIZE, FRAME_SIZE)
            * self.window_energy.clone())
        .sum_dim(2)
            / (3.0 * FFT_SIZE as f32 / 8.0);
        let frames = hops + 1;
        let frame_weights = frame_weights.reshape([batch * frames, 1, 1]);
        let weight_sum = frame_weights.clone().sum();

        let (prediction_magnitude, prediction_complex) =
            self.compressed_spectrum(prediction, right_padding, batch * frames);
        let (target_magnitude, target_complex) =
            self.compressed_spectrum(target, right_padding, batch * frames);
        let magnitude_error = (prediction_magnitude - target_magnitude)
            .square()
            .mean_dim(1);
        let complex_error = (prediction_complex - target_complex)
            .square()
            .mean_dims(&[1, 2]);
        ((magnitude_error + complex_error) * frame_weights).sum() / weight_sum + waveform * 0.1
    }

    fn compressed_spectrum(
        &self,
        waveform: Tensor<2>,
        right_padding: usize,
        total_frames: usize,
    ) -> (Tensor<3>, Tensor<3>) {
        let frames = waveform
            .pad([(FRAME_SIZE, right_padding)], PadMode::Constant(0.0))
            .unfold::<3, _>(1, FFT_SIZE, FRAME_SIZE)
            * self.window.clone();
        // All frames share a single batched FFT invocation per waveform.
        let spectrum =
            self.fft.forward(frames.reshape([total_frames, FFT_SIZE])) / (FFT_SIZE as f32).sqrt();
        debug_assert_eq!(spectrum.dims(), [total_frames, FREQ_BINS, 2]);
        let power = spectrum.clone().square().sum_dim(2) + EPSILON;
        let magnitude = power.clone().powf_scalar(COMPRESSION / 2.0);
        let complex = spectrum * power.powf_scalar((COMPRESSION - 1.0) / 2.0);
        (magnitude, complex)
    }
}

#[cfg(all(test, feature = "flex"))]
mod tests {
    use super::EnhancementLoss;
    use burn::tensor::{Device, Tensor, TensorData};

    fn speech(samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|n| {
                let phase = std::f32::consts::TAU * n as f32 / 73.0;
                0.3 * phase.sin() + 0.1 * (phase * 2.7).cos()
            })
            .collect()
    }

    fn waveform(values: Vec<f32>, batch: usize, samples: usize, device: &Device) -> Tensor<2> {
        Tensor::from_data(TensorData::new(values, [batch, samples]), device)
    }

    fn scalar(loss: Tensor<1>) -> f32 {
        loss.try_into_scalar::<f32>().unwrap()
    }

    // Catches a nonzero offset in compression or an unpaired spectral comparison.
    #[test]
    fn identical_signals_have_zero_loss_for_unequal_valid_lengths() {
        let device = Device::flex();
        let loss = EnhancementLoss::new(&device);
        let input = waveform(speech(2 * 1007), 2, 1007, &device);
        let value = scalar(loss.forward(input.clone(), input, &[1007, 19]));
        assert!(value.abs() < 1e-7, "identical signal loss: {value}");
    }

    // Catches a reversed objective or a loss that ignores its prediction.
    #[test]
    fn moving_prediction_toward_target_reduces_loss() {
        let device = Device::flex();
        let loss = EnhancementLoss::new(&device);
        let target = waveform(speech(721), 1, 721, &device);
        let far = scalar(loss.forward(target.clone() * 0.2, target.clone(), &[721]));
        let close = scalar(loss.forward(target.clone() * 0.8, target, &[721]));
        assert!(close > 0.0 && close < far, "close={close}, far={far}");
    }

    // Magnitude-only objectives cannot distinguish this sign inversion.
    #[test]
    fn phase_inversion_has_more_than_only_waveform_penalty() {
        let device = Device::flex();
        let loss = EnhancementLoss::new(&device);
        let signal = speech(960);
        let waveform_only = 0.1 * signal.iter().map(|x| 2.0 * x.abs()).sum::<f32>() / 960.0;
        let target = waveform(signal, 1, 960, &device);
        let value = scalar(loss.forward(target.clone().neg(), target, &[960]));
        assert!(
            value > waveform_only + 1e-3,
            "phase inversion loss: {value}"
        );
    }

    // Catches masking after FFT, including padded frames, and division by batch capacity.
    #[test]
    fn arbitrary_padding_and_larger_batch_capacity_do_not_change_loss() {
        let device = Device::flex();
        let loss = EnhancementLoss::new(&device);
        let valid = [1, 507];
        let evaluate = |samples: usize, padding: f32| {
            let mut prediction = vec![padding; 2 * samples];
            let mut target = vec![-padding; 2 * samples];
            let signal = speech(507);
            for (batch, &length) in valid.iter().enumerate() {
                for n in 0..length {
                    target[batch * samples + n] = signal[n];
                    prediction[batch * samples + n] = 0.4 * signal[n];
                }
            }
            scalar(loss.forward(
                waveform(prediction, 2, samples, &device),
                waveform(target, 2, samples, &device),
                &valid,
            ))
        };
        let reference = evaluate(1100, 0.0);
        for value in [
            evaluate(1100, 9.0),
            evaluate(1920, -7.0),
            evaluate(1100, f32::NAN),
            evaluate(1100, f32::INFINITY),
            evaluate(1100, f32::NEG_INFINITY),
        ] {
            assert!((value - reference).abs() < 2e-6, "{value} != {reference}");
        }
    }

    #[test]
    fn independent_equal_length_items_have_the_mean_of_their_separate_losses() {
        let device = Device::flex();
        let loss = EnhancementLoss::new(&device);
        let first_target = speech(721);
        let second_target: Vec<_> = first_target.iter().rev().map(|x| x * 0.6).collect();
        let first_prediction: Vec<_> = first_target.iter().map(|x| x * 0.3).collect();
        let second_prediction: Vec<_> = second_target.iter().map(|x| -x * 0.6).collect();
        let first = scalar(loss.forward(
            waveform(first_prediction.clone(), 1, 721, &device),
            waveform(first_target.clone(), 1, 721, &device),
            &[721],
        ));
        let second = scalar(loss.forward(
            waveform(second_prediction.clone(), 1, 721, &device),
            waveform(second_target.clone(), 1, 721, &device),
            &[721],
        ));
        let batched = scalar(loss.forward(
            waveform(
                [first_prediction, second_prediction].concat(),
                2,
                721,
                &device,
            ),
            waveform([first_target, second_target].concat(), 2, 721, &device),
            &[721, 721],
        ));
        assert!((batched - 0.5 * (first + second)).abs() < 2e-6);
    }

    #[test]
    fn unequal_valid_lengths_match_an_independent_impulse_energy_oracle() {
        let device = Device::flex();
        let loss = EnhancementLoss::new(&device);
        let mut prediction = vec![0.0; 2 * 481];
        prediction[0] = 0.1;
        prediction[481] = 0.3;
        let actual = scalar(loss.forward(
            waveform(prediction, 2, 481, &device),
            Tensor::zeros([2, 481], &device),
            &[1, 481],
        ));
        // An impulse at the center of a Hann window has equal FFT-bin power.
        // Valid lengths 1 and 481 give unnormalized frame energies 1 and
        // (180.5, 180.5, 0). Only each item's first frame has a nonzero signal.
        let spectral = |amplitude: f64| {
            let real = amplitude / 960.0f64.sqrt();
            let power = real * real + 1e-8;
            (power.powf(0.15) - 1e-8f64.powf(0.15)).powi(2) + 0.5 * real * real * power.powf(-0.7)
        };
        let expected = (spectral(0.1) + 180.5 * spectral(0.3)) / 362.0 + 0.1 * 0.4 / 482.0;
        assert!(
            (f64::from(actual) - expected).abs() < 2e-6,
            "actual={actual}, expected={expected}"
        );
    }

    // Catches singular powers at zero, disconnected gradients, and unmasked padding gradients.
    #[test]
    fn silence_and_speech_backward_are_finite_and_padding_has_zero_gradient() {
        let device = Device::flex().autodiff();
        let loss = EnhancementLoss::new(&device);
        for (prediction_scale, target_scale, expect_nonzero, padding) in [
            (0.0, 0.0, false, 0.0),
            (0.0, 1.0, true, 0.0),
            (0.5, 1.0, true, 0.0),
            (0.5, 1.0, true, f32::NAN),
            (0.5, 1.0, true, f32::INFINITY),
            (0.5, 1.0, true, f32::NEG_INFINITY),
        ] {
            let signal = speech(601);
            let prediction = waveform(
                signal
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        if i < 503 {
                            x * prediction_scale
                        } else {
                            padding
                        }
                    })
                    .collect(),
                1,
                601,
                &device,
            )
            .require_grad();
            let target = waveform(
                signal
                    .iter()
                    .enumerate()
                    .map(|(i, x)| if i < 503 { x * target_scale } else { -padding })
                    .collect(),
                1,
                601,
                &device,
            );
            let value = loss.forward(prediction.clone(), target, &[503]);
            assert!(scalar(value.clone()).is_finite());
            let gradients = value.backward();
            let gradient = prediction
                .grad(&gradients)
                .expect("loss must be differentiable with respect to prediction")
                .try_into_vec_as::<f32>()
                .unwrap();
            assert!(gradient.iter().all(|x| x.is_finite()));
            assert!(gradient[503..].iter().all(|&x| x == 0.0));
            if expect_nonzero {
                assert!(gradient[..503].iter().any(|x| x.abs() > 1e-7));
            } else {
                assert!(gradient.iter().all(|x| x.abs() < 1e-7));
            }
        }
    }
}
