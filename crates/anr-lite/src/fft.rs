use burn::tensor::{Device, Tensor, TensorData, signal};

use crate::{FFT_SIZE, FREQ_BINS};

const RADIX: usize = 15;
const NATIVE_SIZE: usize = 64;
const NATIVE_BINS: usize = NATIVE_SIZE / 2 + 1;

/// A differentiable 960-point real FFT built from radix-64 and radix-15 stages.
///
/// All stored tensors are fixed device constants, not learnable parameters. The
/// transform is unnormalized in the forward direction and normalized by 1/960
/// in the inverse direction.
#[derive(Clone, Debug)]
pub struct Fft960 {
    device: Device,
    cosine: Tensor<3>,
    sine: Tensor<3>,
    twiddle_cosine: Tensor<3>,
    twiddle_sine: Tensor<3>,
    zero_endpoint: Tensor<2>,
}

impl Fft960 {
    /// Constructs the small DFT and twiddle constants once on `device`.
    pub fn new(device: &Device) -> Self {
        let mut cosine = Vec::with_capacity(RADIX * RADIX);
        let mut sine = Vec::with_capacity(RADIX * RADIX);
        for r in 0..RADIX {
            for q in 0..RADIX {
                let angle = std::f64::consts::TAU * (r * q) as f64 / RADIX as f64;
                cosine.push(angle.cos() as f32);
                sine.push(angle.sin() as f32);
            }
        }
        let mut twiddle_cosine = Vec::with_capacity(RADIX * NATIVE_SIZE);
        let mut twiddle_sine = Vec::with_capacity(RADIX * NATIVE_SIZE);
        for r in 0..RADIX {
            for p in 0..NATIVE_SIZE {
                let angle = std::f64::consts::TAU * (r * p) as f64 / FFT_SIZE as f64;
                twiddle_cosine.push(angle.cos() as f32);
                twiddle_sine.push(angle.sin() as f32);
            }
        }
        // Keep constants outside autodiff, even when the caller supplied an
        // autodiff device. Burn promotes them as constants beside tracked input.
        let constants_device = device.clone().without_autodiff();
        Self {
            device: constants_device.clone(),
            cosine: Tensor::from_data(
                TensorData::new(cosine, [1, RADIX, RADIX]),
                &constants_device,
            ),
            sine: Tensor::from_data(TensorData::new(sine, [1, RADIX, RADIX]), &constants_device),
            twiddle_cosine: Tensor::from_data(
                TensorData::new(twiddle_cosine, [1, RADIX, NATIVE_SIZE]),
                &constants_device,
            ),
            twiddle_sine: Tensor::from_data(
                TensorData::new(twiddle_sine, [1, RADIX, NATIVE_SIZE]),
                &constants_device,
            ),
            zero_endpoint: Tensor::zeros([1, 1], &constants_device),
        }
    }

    /// Transforms `[batch, 960]` real samples into `[batch, 481, 2]` complex bins.
    ///
    /// # Panics
    /// Panics for an empty batch, a wrong sample count, or a different device.
    pub fn forward(&self, input: Tensor<2>) -> Tensor<3> {
        let [batch, samples] = input.dims();
        assert!(
            batch > 0 && samples == FFT_SIZE,
            "Fft960::forward expects [B, 960] with B > 0, got {:?}",
            input.dims()
        );
        assert_eq!(
            input.device(),
            self.device,
            "Fft960::forward input and transform constants must use the same device"
        );

        // x[15*m+r] -> [B,r,m]. The only native FFT has length 64.
        let input = input.reshape([batch, NATIVE_SIZE, RADIX]).swap_dims(1, 2);
        let (half_re, half_im) = signal::rfft(input, 2, Some(NATIVE_SIZE));
        let mirror_re = half_re.clone().narrow(2, 1, NATIVE_SIZE / 2 - 1).flip([2]);
        let mirror_im = half_im
            .clone()
            .narrow(2, 1, NATIVE_SIZE / 2 - 1)
            .flip([2])
            .neg();
        let re = Tensor::cat(vec![half_re, mirror_re], 2);
        let im = Tensor::cat(vec![half_im, mirror_im], 2);

        // U[r,p] = FFT64[r,p] * exp(-i*2*pi*r*p/960).
        let u_re =
            re.clone() * self.twiddle_cosine.clone() + im.clone() * self.twiddle_sine.clone();
        let u_im = im * self.twiddle_cosine.clone() - re * self.twiddle_sine.clone();
        // Matmul over r: [B,p,r] @ [1,r,q], then [B,q,p].
        let u_re = u_re.swap_dims(1, 2);
        let u_im = u_im.swap_dims(1, 2);
        let re = u_re.clone().matmul(self.cosine.clone()) + u_im.clone().matmul(self.sine.clone());
        let im = u_im.matmul(self.cosine.clone()) - u_re.matmul(self.sine.clone());
        let re = re
            .swap_dims(1, 2)
            .reshape([batch, FFT_SIZE])
            .narrow(1, 0, FREQ_BINS);
        let im = im
            .swap_dims(1, 2)
            .reshape([batch, FFT_SIZE])
            .narrow(1, 0, FREQ_BINS);
        let im = self.real_endpoints(im, batch);
        Tensor::stack::<3>(vec![re, im], 2)
    }

    /// Transforms `[batch, 481, 2]` bins into `[batch, 960]` real samples.
    ///
    /// Imaginary DC and Nyquist components are ignored, including nonfinite
    /// values. Interior bins are reflected with Hermitian conjugation.
    ///
    /// # Panics
    /// Panics for an empty batch, a wrong spectrum shape, or a different device.
    pub fn inverse(&self, spectrum: Tensor<3>) -> Tensor<2> {
        let [batch, bins, components] = spectrum.dims();
        assert!(
            batch > 0 && bins == FREQ_BINS && components == 2,
            "Fft960::inverse expects [B, 481, 2] with B > 0, got {:?}",
            spectrum.dims()
        );
        assert_eq!(
            spectrum.device(),
            self.device,
            "Fft960::inverse spectrum and transform constants must use the same device"
        );
        let half_re = spectrum.clone().narrow(2, 0, 1).reshape([batch, FREQ_BINS]);
        let half_im = spectrum.narrow(2, 1, 1).reshape([batch, FREQ_BINS]);
        let half_im = self.real_endpoints(half_im, batch);
        let mirror_re = half_re.clone().narrow(1, 1, FREQ_BINS - 2).flip([1]);
        let mirror_im = half_im.clone().narrow(1, 1, FREQ_BINS - 2).flip([1]).neg();
        // Flattened full spectrum uses k=q*64+p, so unflatten to [B,q,p].
        let re = Tensor::cat(vec![half_re, mirror_re], 1)
            .reshape([batch, RADIX, NATIVE_SIZE])
            .swap_dims(1, 2);
        let im = Tensor::cat(vec![half_im, mirror_im], 1)
            .reshape([batch, RADIX, NATIVE_SIZE])
            .swap_dims(1, 2);
        // Inverse DFT15 over q, explicitly normalized by 1/15.
        let u_re = (re.clone().matmul(self.cosine.clone()) - im.clone().matmul(self.sine.clone()))
            / RADIX as f32;
        let u_im = (im.matmul(self.cosine.clone()) + re.matmul(self.sine.clone())) / RADIX as f32;
        let u_re = u_re.swap_dims(1, 2);
        let u_im = u_im.swap_dims(1, 2);
        // Undo the forward twiddle with the positive complex exponential.
        let re =
            u_re.clone() * self.twiddle_cosine.clone() - u_im.clone() * self.twiddle_sine.clone();
        let im = u_im * self.twiddle_cosine.clone() + u_re * self.twiddle_sine.clone();
        let re = re.narrow(2, 0, NATIVE_BINS);
        let im = im.narrow(2, 0, NATIVE_BINS);
        // Native irfft contributes the remaining 1/64 normalization.
        signal::irfft(re, im, 2, Some(NATIVE_SIZE))
            .swap_dims(1, 2)
            .reshape([batch, FFT_SIZE])
    }

    fn real_endpoints(&self, imaginary: Tensor<2>, batch: usize) -> Tensor<2> {
        let zero = self.zero_endpoint.clone().expand([batch, 1]);
        Tensor::cat(
            vec![zero.clone(), imaginary.narrow(1, 1, FREQ_BINS - 2), zero],
            1,
        )
    }
}

#[cfg(all(test, feature = "flex"))]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn signal(batch: usize) -> Vec<f32> {
        (0..batch * FFT_SIZE)
            .map(|i| {
                let n = (i % FFT_SIZE) as f64;
                let b = (i / FFT_SIZE) as f64;
                ((0.013 * n * n + 0.7 * b).sin() * 0.31
                    + (0.17 * n + b).cos() * 0.23
                    + ((i * 37 % 101) as f64 - 50.0) * 0.003) as f32
            })
            .collect()
    }

    fn assert_close(actual: f32, expected: f64, tolerance: f64) {
        assert!(
            (actual as f64 - expected).abs() <= tolerance,
            "actual {actual}, expected {expected}, tolerance {tolerance}"
        );
    }

    // Catches wrong mixed-radix ordering, twiddle signs and batch broadcasting.
    #[test]
    fn forward_matches_independent_selected_bin_scalar_dft() {
        let device = Device::flex();
        let fft = Fft960::new(&device);
        let values = signal(2);
        let output = fft.forward(Tensor::from_data(
            TensorData::new(values.clone(), [2, FFT_SIZE]),
            &device,
        ));
        assert_eq!(output.dims(), [2, FREQ_BINS, 2]);
        let output = output.try_into_vec_as::<f32>().unwrap();
        for batch in 0..2 {
            for k in [0, 1, 14, 15, 31, 32, 63, 64, 65, 127, 251, 479, 480] {
                let mut real = 0.0;
                let mut imag = 0.0;
                for n in 0..FFT_SIZE {
                    let angle = TAU * (k * n) as f64 / FFT_SIZE as f64;
                    let x = values[batch * FFT_SIZE + n] as f64;
                    real += x * angle.cos();
                    imag -= x * angle.sin();
                }
                let offset = (batch * FREQ_BINS + k) * 2;
                assert_close(output[offset], real, 4e-3);
                assert_close(output[offset + 1], imag, 4e-3);
            }
        }
    }

    // Independent analytic inverse fixture prevents forward/inverse errors cancelling.
    #[test]
    fn inverse_reconstructs_known_bins_and_ignores_endpoint_imaginary_parts() {
        let device = Device::flex();
        let fft = Fft960::new(&device);
        let mut bins = vec![0.0f32; FREQ_BINS * 2];
        bins[0] = 0.21 * FFT_SIZE as f32;
        bins[1] = f32::NAN;
        bins[480 * 2] = -0.08 * FFT_SIZE as f32;
        bins[480 * 2 + 1] = f32::INFINITY;
        bins[137 * 2] = (FFT_SIZE as f64 * 0.3 * 0.3f64.cos()) as f32;
        bins[137 * 2 + 1] = (FFT_SIZE as f64 * 0.3 * 0.3f64.sin()) as f32;
        bins[63 * 2 + 1] = -0.2 * FFT_SIZE as f32;
        let output = fft.inverse(Tensor::from_data(
            TensorData::new(bins, [1, FREQ_BINS, 2]),
            &device,
        ));
        assert_eq!(output.dims(), [1, FFT_SIZE]);
        let output = output.try_into_vec_as::<f32>().unwrap();
        for (n, actual) in output.into_iter().enumerate() {
            let expected = 0.21
                + 0.6 * (TAU * 137.0 * n as f64 / FFT_SIZE as f64 + 0.3).cos()
                + 0.4 * (TAU * 63.0 * n as f64 / FFT_SIZE as f64).sin()
                - 0.08 * if n % 2 == 0 { 1.0 } else { -1.0 };
            assert_close(actual, expected, 2e-5);
        }
    }

    #[test]
    fn impulse_has_correct_phase_at_nontrivial_bins() {
        let device = Device::flex();
        let mut input = vec![0.0f32; FFT_SIZE];
        input[37] = 1.0;
        let result = Fft960::new(&device)
            .forward(Tensor::from_data(
                TensorData::new(input, [1, FFT_SIZE]),
                &device,
            ))
            .try_into_vec_as::<f32>()
            .unwrap();
        for k in [0, 17, 63, 64, 137, 480] {
            let angle = TAU * (37 * k) as f64 / FFT_SIZE as f64;
            assert_close(result[2 * k], angle.cos(), 2e-6);
            assert_close(result[2 * k + 1], -angle.sin(), 2e-6);
        }
    }

    #[test]
    fn roundtrip_and_single_batch_results_are_independent() {
        let device = Device::flex();
        let fft = Fft960::new(&device);
        let values = signal(3);
        let input = Tensor::from_data(TensorData::new(values.clone(), [3, FFT_SIZE]), &device);
        let spectrum = fft.forward(input);
        let output = fft
            .inverse(spectrum.clone())
            .try_into_vec_as::<f32>()
            .unwrap();
        for (actual, expected) in output.iter().zip(&values) {
            assert_close(*actual, *expected as f64, 2e-5);
        }
        let batched = spectrum.clone().try_into_vec_as::<f32>().unwrap();
        for b in 0..3 {
            let solo_input = Tensor::from_data(
                TensorData::new(
                    values[b * FFT_SIZE..(b + 1) * FFT_SIZE].to_vec(),
                    [1, FFT_SIZE],
                ),
                &device,
            );
            let solo = fft.forward(solo_input).try_into_vec_as::<f32>().unwrap();
            for (actual, expected) in solo
                .iter()
                .zip(&batched[b * FREQ_BINS * 2..(b + 1) * FREQ_BINS * 2])
            {
                assert_close(*actual, *expected as f64, 2e-5);
            }
            let solo_inverse = fft
                .inverse(spectrum.clone().narrow(0, b, 1))
                .try_into_vec_as::<f32>()
                .unwrap();
            for (actual, expected) in solo_inverse
                .iter()
                .zip(&output[b * FFT_SIZE..(b + 1) * FFT_SIZE])
            {
                assert_close(*actual, *expected as f64, 2e-5);
            }
        }
    }

    // Uses a single independently known Fourier basis, not FFT followed by inverse FFT.
    #[test]
    fn forward_gradient_matches_analytic_frequency_basis() {
        let device = Device::flex().autodiff();
        let fft = Fft960::new(&device);
        let input =
            Tensor::from_data(TensorData::new(signal(2), [2, FFT_SIZE]), &device).require_grad();
        let bin = fft.forward(input.clone()).narrow(0, 1, 1).narrow(1, 137, 1);
        let loss = bin.clone().narrow(2, 0, 1).sum() * 0.7 + bin.narrow(2, 1, 1).sum() * 1.3;
        let grads = loss.backward();
        let gradient = input
            .grad(&grads)
            .unwrap()
            .try_into_vec_as::<f32>()
            .unwrap();
        for n in 0..FFT_SIZE {
            assert_close(gradient[n], 0.0, 2e-5);
            let angle = TAU * 137.0 * n as f64 / FFT_SIZE as f64;
            assert_close(
                gradient[FFT_SIZE + n],
                0.7 * angle.cos() - 1.3 * angle.sin(),
                2e-5,
            );
        }
    }

    #[test]
    fn inverse_gradient_matches_analytic_real_signal_basis() {
        let device = Device::flex().autodiff();
        let fft = Fft960::new(&device);
        let spectrum = Tensor::<3>::zeros([1, FREQ_BINS, 2], &device).require_grad();
        let loss = fft.inverse(spectrum.clone()).narrow(1, 37, 1).sum();
        let grads = loss.backward();
        let gradient = spectrum
            .grad(&grads)
            .unwrap()
            .try_into_vec_as::<f32>()
            .unwrap();
        for k in [1, 63, 64, 137, 479] {
            let angle = TAU * (k * 37) as f64 / FFT_SIZE as f64;
            assert_close(gradient[k * 2], 2.0 * angle.cos() / FFT_SIZE as f64, 1e-6);
            assert_close(
                gradient[k * 2 + 1],
                -2.0 * angle.sin() / FFT_SIZE as f64,
                1e-6,
            );
        }
        assert_close(gradient[0], 1.0 / FFT_SIZE as f64, 1e-6);
        assert_close(gradient[1], 0.0, 1e-7);
        assert_close(gradient[480 * 2], -1.0 / FFT_SIZE as f64, 1e-6);
        assert_close(gradient[480 * 2 + 1], 0.0, 1e-7);
    }

    #[test]
    #[should_panic]
    fn inverse_rejects_wrong_spectrum_layout() {
        let device = Device::flex();
        Fft960::new(&device).inverse(Tensor::zeros([2, 481, 3], &device));
    }
}
