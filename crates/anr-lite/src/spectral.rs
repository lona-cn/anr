//! Sparse ERB analysis and causal complex residual filtering.

use burn::tensor::{Device, IndexingUpdateOp, Int, Tensor, TensorData};

use crate::{DF_BINS, DF_TAPS, ERB_BANDS, FFT_SIZE, FREQ_BINS, LOOKAHEAD_FRAMES, SAMPLE_RATE};

const FULL_HISTORY: usize = LOOKAHEAD_FRAMES + 1;
const POWER_EPSILON: f32 = 1e-12;

/// Per-stream differentiable feature statistics and spectral histories.
///
/// Histories are oldest first. Feature extraction updates only the EMA; calling
/// [`SpectralProcessor::filter`] advances both spectral histories exactly once.
#[derive(Clone, Debug)]
pub struct SpectralState {
    ema: Tensor<2>,
    full_history: Tensor<4>,
    low_history: Tensor<4>,
}

impl SpectralState {
    /// Disconnect all recurrent state from the preceding autograd graph.
    pub fn detach(self) -> Self {
        Self {
            ema: self.ema.detach(),
            full_history: self.full_history.detach(),
            low_history: self.low_history.detach(),
        }
    }
}

/// Device-resident sparse ERB interpolation and five-tap residual filtering.
///
/// Each frequency bin contributes to at most two ERB bands. Analysis weights
/// are normalized independently per band; synthesis weights form a partition
/// of unity. No dense frequency-by-band matrix is constructed.
#[derive(Clone, Debug)]
pub struct SpectralProcessor {
    device: Device,
    bin_indices: Tensor<1, Int>,
    band_indices: Tensor<1, Int>,
    analysis_weights: Tensor<2>,
    interpolation_weights: Tensor<2>,
    taper: Tensor<3>,
    real_endpoints: Tensor<3>,
}

impl SpectralProcessor {
    /// Precompute the fixed sparse ERB map and endpoint/taper masks on `device`.
    pub fn new(device: &Device) -> Self {
        let centers = erb_centers();
        let mut bin_indices = Vec::with_capacity(2 * FREQ_BINS);
        let mut band_indices = Vec::with_capacity(2 * FREQ_BINS);
        let mut weights = Vec::with_capacity(2 * FREQ_BINS);
        let mut coverage = [0.0f32; ERB_BANDS];
        let mut lower = 0;
        for bin in 0..FREQ_BINS {
            while lower + 1 < ERB_BANDS - 1 && bin > centers[lower + 1] {
                lower += 1;
            }
            let upper_weight =
                (bin - centers[lower]) as f32 / (centers[lower + 1] - centers[lower]) as f32;
            for (band, weight) in [(lower, 1.0 - upper_weight), (lower + 1, upper_weight)] {
                if weight > 0.0 {
                    bin_indices.push(bin as i64);
                    band_indices.push(band as i64);
                    weights.push(weight);
                    coverage[band] += weight;
                }
            }
        }
        assert!(coverage.iter().all(|weight| *weight > 0.0));
        let analysis: Vec<f32> = weights
            .iter()
            .zip(&band_indices)
            .map(|(&weight, &band)| weight / coverage[band as usize])
            .collect();
        let edges = weights.len();
        let taper: Vec<f32> = (0..DF_BINS)
            .map(|bin| {
                if bin < 80 {
                    1.0
                } else {
                    0.5 * (1.0 + (std::f32::consts::PI * (bin - 80) as f32 / 15.0).cos())
                }
            })
            .collect();
        let mut endpoints = vec![1.0f32; FREQ_BINS * 2];
        endpoints[1] = 0.0;
        endpoints[(FREQ_BINS - 1) * 2 + 1] = 0.0;
        Self {
            device: device.clone(),
            bin_indices: Tensor::from_data(TensorData::new(bin_indices, [edges]), device),
            band_indices: Tensor::from_data(TensorData::new(band_indices, [edges]), device),
            analysis_weights: Tensor::from_data(TensorData::new(analysis, [1, edges]), device),
            interpolation_weights: Tensor::from_data(TensorData::new(weights, [1, edges]), device),
            taper: Tensor::from_data(TensorData::new(taper, [1, DF_BINS, 1]), device),
            real_endpoints: Tensor::from_data(
                TensorData::new(endpoints, [1, FREQ_BINS, 2]),
                device,
            ),
        }
    }

    /// Create zero-initialized state for a nonempty batch on this processor's device.
    pub fn initial_state(&self, batch: usize) -> SpectralState {
        assert!(batch > 0, "spectral batch must be nonzero");
        SpectralState {
            ema: Tensor::zeros([batch, ERB_BANDS], &self.device),
            full_history: Tensor::zeros([batch, FULL_HISTORY, FREQ_BINS, 2], &self.device),
            low_history: Tensor::zeros([batch, DF_TAPS, DF_BINS, 2], &self.device),
        }
    }

    /// Extract `[B,64]` ERB features and `[B,3,96]` fine features.
    ///
    /// Input must be `[B,481,2]` with `B > 0`, matching the state batch and the
    /// processor device. This updates only the running EMA, retaining its graph.
    pub fn features(
        &self,
        spectrum: Tensor<3>,
        state: &mut SpectralState,
    ) -> (Tensor<2>, Tensor<3>) {
        let batch = self.check_input(&spectrum, state);
        let re = spectrum
            .clone()
            .slice([0..batch, 0..FREQ_BINS, 0..1])
            .squeeze_dim::<2>(2);
        let im = spectrum
            .slice([0..batch, 0..FREQ_BINS, 1..2])
            .squeeze_dim::<2>(2);
        let power = re.clone() * re.clone() + im.clone() * im.clone();
        let band_db = (self.analyze_power(power.clone()) + POWER_EPSILON).log()
            * (10.0f32 / std::f32::consts::LN_10);
        state.ema = state.ema.clone() * 0.99 + band_db.clone() * 0.01;
        let coarse = (band_db - state.ema.clone()) / 40.0;
        let magnitude = (power.slice([0..batch, 0..DF_BINS]) + POWER_EPSILON).sqrt();
        let denominator = magnitude.clone().powf_scalar(0.7);
        let fine = Tensor::cat(
            vec![
                magnitude.log().unsqueeze_dim::<3>(1),
                (re.slice([0..batch, 0..DF_BINS]) / denominator.clone()).unsqueeze_dim::<3>(1),
                (im.slice([0..batch, 0..DF_BINS]) / denominator).unsqueeze_dim::<3>(1),
            ],
            1,
        );
        (coarse, fine)
    }

    /// Enhance target frame `t-2`, advancing the histories with current frame `t`.
    ///
    /// Shapes are spectrum `[B,481,2]`, gains `[B,64]`, and complex residual
    /// coefficients `[B,5,96,2]`, all on the processor device. Tap zero multiplies
    /// the current spectrum; tap four multiplies frame `t-4`. DC and Nyquist
    /// imaginary output components are always zero.
    pub fn filter(
        &self,
        spectrum: Tensor<3>,
        gains: Tensor<2>,
        coefficients: Tensor<4>,
        state: &mut SpectralState,
    ) -> Tensor<3> {
        let batch = self.check_input(&spectrum, state);
        assert_eq!(
            gains.dims(),
            [batch, ERB_BANDS],
            "gain shape must be [B,64]"
        );
        assert_eq!(
            coefficients.dims(),
            [batch, DF_TAPS, DF_BINS, 2],
            "coefficient shape must be [B,5,96,2]"
        );
        assert_eq!(gains.device(), self.device, "gain device mismatch");
        assert_eq!(
            coefficients.device(),
            self.device,
            "coefficient device mismatch"
        );
        state.full_history = Tensor::cat(
            vec![
                state
                    .full_history
                    .clone()
                    .slice([0..batch, 1..FULL_HISTORY, 0..FREQ_BINS, 0..2]),
                spectrum.clone().unsqueeze_dim::<4>(1),
            ],
            1,
        );
        state.low_history = Tensor::cat(
            vec![
                state
                    .low_history
                    .clone()
                    .slice([0..batch, 1..DF_TAPS, 0..DF_BINS, 0..2]),
                spectrum
                    .slice([0..batch, 0..DF_BINS, 0..2])
                    .unsqueeze_dim::<4>(1),
            ],
            1,
        );
        let target = state
            .full_history
            .clone()
            .slice([0..batch, 0..1, 0..FREQ_BINS, 0..2])
            .squeeze_dim::<3>(1);
        let base = target * self.interpolate_gains(gains).unsqueeze_dim::<3>(2);
        let history = state.low_history.clone().flip([1]);
        let xr = history
            .clone()
            .slice([0..batch, 0..DF_TAPS, 0..DF_BINS, 0..1])
            .squeeze_dim::<3>(3);
        let xi = history
            .slice([0..batch, 0..DF_TAPS, 0..DF_BINS, 1..2])
            .squeeze_dim::<3>(3);
        let cr = coefficients
            .clone()
            .slice([0..batch, 0..DF_TAPS, 0..DF_BINS, 0..1])
            .squeeze_dim::<3>(3);
        let ci = coefficients
            .slice([0..batch, 0..DF_TAPS, 0..DF_BINS, 1..2])
            .squeeze_dim::<3>(3);
        let real = (cr.clone() * xr.clone() - ci.clone() * xi.clone())
            .sum_dim(1)
            .squeeze_dim::<2>(1);
        let imaginary = (cr * xi + ci * xr).sum_dim(1).squeeze_dim::<2>(1);
        let residual = Tensor::cat(
            vec![real.unsqueeze_dim::<3>(2), imaginary.unsqueeze_dim::<3>(2)],
            2,
        ) * self.taper.clone();
        let residual = Tensor::cat(
            vec![
                residual,
                Tensor::zeros([batch, FREQ_BINS - DF_BINS, 2], &self.device),
            ],
            1,
        );
        (base + residual) * self.real_endpoints.clone()
    }

    fn check_input(&self, spectrum: &Tensor<3>, state: &SpectralState) -> usize {
        let [batch, bins, complex] = spectrum.dims();
        assert!(batch > 0, "spectral batch must be nonzero");
        assert_eq!(
            [bins, complex],
            [FREQ_BINS, 2],
            "spectrum shape must be [B,481,2]"
        );
        assert_eq!(spectrum.device(), self.device, "spectrum device mismatch");
        assert_eq!(
            state.ema.dims(),
            [batch, ERB_BANDS],
            "EMA state batch mismatch"
        );
        assert_eq!(
            state.full_history.dims(),
            [batch, FULL_HISTORY, FREQ_BINS, 2],
            "full history shape mismatch"
        );
        assert_eq!(
            state.low_history.dims(),
            [batch, DF_TAPS, DF_BINS, 2],
            "low history shape mismatch"
        );
        assert_eq!(state.ema.device(), self.device, "EMA state device mismatch");
        assert_eq!(
            state.full_history.device(),
            self.device,
            "full history device mismatch"
        );
        assert_eq!(
            state.low_history.device(),
            self.device,
            "low history device mismatch"
        );
        batch
    }

    fn analyze_power(&self, power: Tensor<2>) -> Tensor<2> {
        let batch = power.dims()[0];
        let contributions =
            power.select(1, self.bin_indices.clone()) * self.analysis_weights.clone();
        // Burn 0.22 calls select-add `select_assign(..., Add)`. Duplicate band
        // indices accumulate; backward gathers each contribution's gradient.
        Tensor::zeros([batch, ERB_BANDS], &self.device).select_assign(
            1,
            self.band_indices.clone(),
            contributions,
            IndexingUpdateOp::Add,
        )
    }

    fn interpolate_gains(&self, gains: Tensor<2>) -> Tensor<2> {
        let batch = gains.dims()[0];
        let contributions =
            gains.select(1, self.band_indices.clone()) * self.interpolation_weights.clone();
        Tensor::zeros([batch, FREQ_BINS], &self.device).select_assign(
            1,
            self.bin_indices.clone(),
            contributions,
            IndexingUpdateOp::Add,
        )
    }
}

fn erb_centers() -> [usize; ERB_BANDS] {
    let mut centers = [0; ERB_BANDS];
    let nyquist = SAMPLE_RATE as f64 / 2.0;
    let max_erb = (1.0 + 0.00437 * nyquist).ln();
    let bin_hz = SAMPLE_RATE as f64 / FFT_SIZE as f64;
    for band in 1..ERB_BANDS {
        let hz = ((max_erb * band as f64 / (ERB_BANDS - 1) as f64).exp() - 1.0) / 0.00437;
        let desired = (hz / bin_hz).round() as usize;
        let minimum = centers[band - 1] + 1;
        let maximum = FREQ_BINS - 1 - (ERB_BANDS - 1 - band);
        centers[band] = desired.clamp(minimum, maximum);
    }
    centers[ERB_BANDS - 1] = FREQ_BINS - 1;
    centers
}

#[cfg(all(test, feature = "flex"))]
mod tests {
    use super::*;

    fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
        tensor.into_data().try_to_vec::<f32>().unwrap()
    }

    fn spectrum(frame: usize, batch: usize, device: &Device) -> Tensor<3> {
        let mut data = Vec::with_capacity(batch * FREQ_BINS * 2);
        for b in 0..batch {
            for bin in 0..FREQ_BINS {
                let x = (frame + 1) as f32 * (b + 1) as f32;
                data.push(x);
                data.push(if bin == 0 || bin == FREQ_BINS - 1 {
                    0.0
                } else {
                    2.0 * x
                });
            }
        }
        Tensor::from_data(TensorData::new(data, [batch, FREQ_BINS, 2]), device)
    }

    fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() <= tolerance, "index {i}: {a} != {e}");
        }
    }

    #[test]
    fn unity_gains_preserve_exact_two_frame_delay_and_batch_independence() {
        let device = Device::flex();
        let processor = SpectralProcessor::new(&device);
        let mut state = processor.initial_state(2);
        for t in 0..7 {
            let current = spectrum(t, 2, &device);
            // Feature extraction must not advance either spectral history.
            let _ = processor.features(current.clone(), &mut state);
            let output = processor.filter(
                current,
                Tensor::ones([2, ERB_BANDS], &device),
                Tensor::zeros([2, DF_TAPS, DF_BINS, 2], &device),
                &mut state,
            );
            let expected = if t < 2 {
                Tensor::zeros([2, FREQ_BINS, 2], &device)
            } else {
                spectrum(t - 2, 2, &device)
            };
            close(&values(output), &values(expected), 1e-6);
        }
    }

    #[test]
    fn complex_residual_uses_current_first_taps_and_cosine_taper() {
        let device = Device::flex();
        let processor = SpectralProcessor::new(&device);
        let mut state = processor.initial_state(2);
        let mut coefficients = vec![0.0; 2 * DF_TAPS * DF_BINS * 2];
        for b in 0..2 {
            for bin in 0..DF_BINS {
                coefficients[((b * DF_TAPS) * DF_BINS + bin) * 2 + 1] = 1.0;
                coefficients[((b * DF_TAPS + 4) * DF_BINS + bin) * 2] = 2.0;
                coefficients[((b * DF_TAPS + 4) * DF_BINS + bin) * 2 + 1] = -0.5;
            }
        }
        let coefficients = Tensor::from_data(
            TensorData::new(coefficients, [2, DF_TAPS, DF_BINS, 2]),
            &device,
        );
        let mut output = Tensor::zeros([2, FREQ_BINS, 2], &device);
        for t in 0..5 {
            output = processor.filter(
                spectrum(t, 2, &device),
                Tensor::zeros([2, ERB_BANDS], &device),
                coefficients.clone(),
                &mut state,
            );
        }
        let data = values(output);
        for b in 0..2 {
            let scale = (b + 1) as f32;
            for bin in 1..FREQ_BINS {
                let taper = if bin < 80 {
                    1.0
                } else if bin < 96 {
                    0.5 * (1.0 + (std::f32::consts::PI * (bin - 80) as f32 / 15.0).cos())
                } else {
                    0.0
                };
                let offset = (b * FREQ_BINS + bin) * 2;
                close(
                    &data[offset..offset + 2],
                    &[-7.0 * scale * taper, 8.5 * scale * taper],
                    2e-5,
                );
            }
            assert_eq!(data[b * FREQ_BINS * 2 + 1], 0.0);
        }
    }

    #[test]
    fn erb_sparse_maps_cover_every_band_and_preserve_constant_power_and_gains() {
        let device = Device::flex();
        let processor = SpectralProcessor::new(&device);
        let centers = erb_centers();
        assert_eq!(centers[0], 0);
        assert_eq!(centers[ERB_BANDS - 1], FREQ_BINS - 1);
        assert!(centers.windows(2).all(|pair| pair[0] < pair[1]));
        let band_power = processor.analyze_power(Tensor::ones([2, FREQ_BINS], &device) * 3.0);
        close(&values(band_power), &vec![3.0; 2 * ERB_BANDS], 3e-6);
        let gains = processor.interpolate_gains(Tensor::ones([2, ERB_BANDS], &device) * 0.375);
        close(&values(gains), &vec![0.375; 2 * FREQ_BINS], 1e-7);
        // Every band is observable at its center, independently of neighboring bands.
        let distinct: Vec<f32> = (0..ERB_BANDS).map(|i| (i + 1) as f32).collect();
        let interpolated = values(processor.interpolate_gains(Tensor::from_data(
            TensorData::new(distinct.clone(), [1, ERB_BANDS]),
            &device,
        )));
        for (band, bin) in centers.iter().enumerate() {
            assert_eq!(interpolated[*bin], distinct[band]);
        }
    }

    #[test]
    fn silence_features_have_defined_values_and_finite_gradients() {
        let device = Device::flex().autodiff();
        let processor = SpectralProcessor::new(&device);
        let mut state = processor.initial_state(2);
        let input = Tensor::zeros([2, FREQ_BINS, 2], &device).require_grad();
        let (coarse, fine) = processor.features(input.clone(), &mut state);
        close(&values(coarse.clone()), &vec![-2.97; 2 * ERB_BANDS], 2e-5);
        let fine_data = values(fine.clone());
        for b in 0..2 {
            close(
                &fine_data[b * 3 * DF_BINS..(b * 3 + 1) * DF_BINS],
                &vec![(1e-6f32).ln(); DF_BINS],
                1e-5,
            );
            close(
                &fine_data[(b * 3 + 1) * DF_BINS..(b + 1) * 3 * DF_BINS],
                &vec![0.0; 2 * DF_BINS],
                0.0,
            );
        }
        let gradients = (coarse.sum() + fine.sum()).backward();
        let gradient = values(input.grad(&gradients).expect("feature gradient"));
        assert!(gradient.iter().all(|value| value.is_finite()));
        // Real and imaginary normalized components have a positive derivative at zero.
        assert!((gradient[2] - 1e-6f32.powf(-0.7)).abs() < 0.1);
        assert!((gradient[3] - 1e-6f32.powf(-0.7)).abs() < 0.1);
        assert_eq!(gradient[2 * DF_BINS], 0.0);
    }

    #[test]
    fn recurrent_ema_and_filter_history_retain_gradients_until_detached() {
        let device = Device::flex().autodiff();
        let processor = SpectralProcessor::new(&device);
        let mut state = processor.initial_state(1);
        let first = spectrum(0, 1, &device).require_grad();
        let _ = processor.features(first.clone(), &mut state);
        let _ = processor.filter(
            first.clone(),
            Tensor::ones([1, ERB_BANDS], &device),
            Tensor::zeros([1, DF_TAPS, DF_BINS, 2], &device),
            &mut state,
        );
        let (features, _) = processor.features(spectrum(1, 1, &device), &mut state);
        let gradients = features.sum().backward();
        let gradient = values(first.grad(&gradients).expect("EMA history gradient"));
        assert!(gradient.iter().any(|value| value.abs() > 1e-6));
        let mut detached = state.detach();
        let fresh = spectrum(2, 1, &device).require_grad();
        let (coarse, fine) = processor.features(fresh.clone(), &mut detached);
        let _ = processor.filter(
            fresh.clone(),
            Tensor::ones([1, ERB_BANDS], &device),
            Tensor::ones([1, DF_TAPS, DF_BINS, 2], &device),
            &mut detached,
        );
        // The original first frame is now the delayed full-band target and
        // also tap two of the low-band history, so both detached histories
        // participate in this loss alongside the detached EMA.
        let output = processor.filter(
            spectrum(3, 1, &device),
            Tensor::ones([1, ERB_BANDS], &device),
            Tensor::ones([1, DF_TAPS, DF_BINS, 2], &device),
            &mut detached,
        );
        let gradients = (coarse.sum() + fine.sum() + output.sum()).backward();
        assert!(
            first.grad(&gradients).is_none(),
            "detach must block all old-state paths"
        );
        let gradient = values(fresh.grad(&gradients).expect("fresh input gradient"));
        assert!(gradient.iter().all(|value| value.is_finite()));
        assert!(gradient.iter().any(|value| value.abs() > 1e-6));
    }

    #[test]
    fn residual_does_not_modify_high_frequency_gain_path_and_endpoints_are_real() {
        let device = Device::flex();
        let processor = SpectralProcessor::new(&device);
        let mut state = processor.initial_state(1);
        let mut output = Tensor::zeros([1, FREQ_BINS, 2], &device);
        for t in 0..3 {
            // Intentionally include imaginary endpoint inputs to exercise the mask.
            let input = spectrum(t, 1, &device) + 1.0;
            output = processor.filter(
                input,
                Tensor::ones([1, ERB_BANDS], &device) * 0.7,
                Tensor::ones([1, DF_TAPS, DF_BINS, 2], &device),
                &mut state,
            );
        }
        let data = values(output);
        for bin in DF_BINS..FREQ_BINS - 1 {
            close(&data[bin * 2..bin * 2 + 2], &[1.4, 2.1], 2e-6);
        }
        close(&data[(FREQ_BINS - 1) * 2..], &[1.4, 0.0], 2e-6);
        assert_eq!(data[1], 0.0);
    }

    #[test]
    fn delayed_base_and_complex_taps_backpropagate_through_state() {
        let device = Device::flex().autodiff();
        let processor = SpectralProcessor::new(&device);
        let mut state = processor.initial_state(1);
        let first = spectrum(0, 1, &device).require_grad();
        let _ = processor.filter(
            first.clone(),
            Tensor::ones([1, ERB_BANDS], &device),
            Tensor::zeros([1, DF_TAPS, DF_BINS, 2], &device),
            &mut state,
        );
        let _ = processor.filter(
            spectrum(1, 1, &device),
            Tensor::ones([1, ERB_BANDS], &device),
            Tensor::zeros([1, DF_TAPS, DF_BINS, 2], &device),
            &mut state,
        );
        let mut coefficients = vec![0.0f32; DF_TAPS * DF_BINS * 2];
        // At t=2, tap 2 references the first frame: i*(re+i*im).
        coefficients[(2 * DF_BINS + 10) * 2 + 1] = 2.0;
        let coefficients = Tensor::from_data(
            TensorData::new(coefficients, [1, DF_TAPS, DF_BINS, 2]),
            &device,
        )
        .require_grad();
        let output = processor.filter(
            spectrum(2, 1, &device),
            Tensor::ones([1, ERB_BANDS], &device),
            coefficients.clone(),
            &mut state,
        );
        let gradients = output.sum().backward();
        let input_gradient = values(first.grad(&gradients).unwrap());
        close(&input_gradient[20..22], &[3.0, -1.0], 2e-6);
        close(&input_gradient[200..202], &[1.0, 1.0], 2e-6);
        let coefficient_gradient = values(coefficients.grad(&gradients).unwrap());
        close(
            &coefficient_gradient[(2 * DF_BINS + 10) * 2..(2 * DF_BINS + 10) * 2 + 2],
            &[3.0, -1.0],
            2e-6,
        );
    }
}
