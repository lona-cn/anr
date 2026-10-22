//! The ANR-Lite-48 frame network and explicit differentiable stream state.

use burn::module::Module;
use burn::nn::{
    Initializer,
    conv::{Conv2d, Conv2dConfig},
};
use burn::tensor::{
    Device, Tensor,
    activation::{relu, sigmoid},
};

use crate::layers::{GroupedLinear, RecurrentBlock, SeparableBlock, Stem};
use crate::{DF_BINS, DF_TAPS, ERB_BANDS};

/// Network predictions for a single spectral frame.
#[derive(Clone, Debug)]
pub struct ModelOutput {
    /// Full-band perceptual gains, `[batch, 64]`.
    pub gains: Tensor<2>,
    /// Complex residual coefficients, `[batch, 5, 96, 2]`, real then imaginary.
    pub coefficients: Tensor<4>,
}

/// Causal raw feature histories and the two *unmixed* GRU hidden states.
///
/// History is stored oldest first. None of the frame updates detach tensors;
/// call [`Self::detach`] explicitly at a truncated-backpropagation boundary.
#[derive(Clone, Debug)]
pub struct ModelState {
    /// Last two coarse input frames, `[batch, 1, 2, 64]`.
    pub coarse_history: Tensor<4>,
    /// Last four fine input frames, `[batch, 3, 4, 96]`.
    pub fine_history: Tensor<4>,
    /// Separate recurrent hidden states, each `[batch, 128]`.
    pub hidden: [Tensor<2>; 2],
}

impl ModelState {
    /// Creates a zero-initialized stream for a strictly positive batch size.
    pub fn new(batch: usize, device: &Device) -> Self {
        assert!(batch > 0, "model state batch must be nonzero");
        Self {
            coarse_history: Tensor::zeros([batch, 1, 2, ERB_BANDS], device),
            fine_history: Tensor::zeros([batch, 3, 4, DF_BINS], device),
            hidden: [
                Tensor::zeros([batch, 128], device),
                Tensor::zeros([batch, 128], device),
            ],
        }
    }

    fn validate(&self, batch: usize, device: &Device) {
        assert!(batch > 0, "model state batch must be nonzero");
        assert_eq!(
            self.coarse_history.dims(),
            [batch, 1, 2, ERB_BANDS],
            "coarse history shape"
        );
        assert_eq!(
            self.fine_history.dims(),
            [batch, 3, 4, DF_BINS],
            "fine history shape"
        );
        assert_eq!(
            self.hidden[0].dims(),
            [batch, 128],
            "first GRU hidden shape"
        );
        assert_eq!(
            self.hidden[1].dims(),
            [batch, 128],
            "second GRU hidden shape"
        );
        assert_eq!(
            self.coarse_history.device(),
            *device,
            "coarse history device"
        );
        assert_eq!(self.fine_history.device(), *device, "fine history device");
        assert_eq!(self.hidden[0].device(), *device, "first GRU hidden device");
        assert_eq!(self.hidden[1].device(), *device, "second GRU hidden device");
    }

    /// Breaks all historical autograd paths without changing the stream values.
    ///
    /// Malformed state is rejected rather than silently normalized.
    pub fn detach(self) -> Self {
        self.validate(self.coarse_history.dims()[0], &self.coarse_history.device());
        Self {
            coarse_history: self.coarse_history.detach(),
            fine_history: self.fine_history.detach(),
            hidden: self.hidden.map(Tensor::detach),
        }
    }
}

/// Compact two-branch causal network, with native Burn records and mode changes.
///
/// The constructor follows the supplied device. Use Burn [`Module::train`] and
/// [`Module::valid`] to select parameter/autodiff and batch-normalization modes.
#[derive(Module, Debug)]
pub struct AnrLite48 {
    coarse_stem: Stem,
    fine_stem: Stem,
    coarse_down_1: SeparableBlock,
    coarse_down_2: SeparableBlock,
    fine_down: SeparableBlock,
    joint_projection: GroupedLinear,
    recurrent_1: RecurrentBlock,
    recurrent_2: RecurrentBlock,
    decoder_projection: GroupedLinear,
    coarse_up_1: SeparableBlock,
    coarse_up_2: SeparableBlock,
    fine_up: SeparableBlock,
    gain_head: Conv2d,
    filter_head: Conv2d,
}

impl AnrLite48 {
    /// Initializes real, trainable layers; no pretrained weights are included.
    pub fn new(device: &Device) -> Self {
        Self {
            coarse_stem: Stem::new(1, 3, device),
            fine_stem: Stem::new(3, 5, device),
            coarse_down_1: SeparableBlock::new(2, device),
            coarse_down_2: SeparableBlock::new(2, device),
            fine_down: SeparableBlock::new(2, device),
            joint_projection: GroupedLinear::new(1024, 128, 8, device),
            recurrent_1: RecurrentBlock::new(device),
            recurrent_2: RecurrentBlock::new(device),
            decoder_projection: GroupedLinear::new(128, 1024, 8, device),
            coarse_up_1: SeparableBlock::new(1, device),
            coarse_up_2: SeparableBlock::new(1, device),
            fine_up: SeparableBlock::new(1, device),
            gain_head: Conv2dConfig::new([16, 1], [1, 1])
                .with_bias(true)
                .init(device),
            filter_head: Conv2dConfig::new([16, DF_TAPS * 2], [1, 1])
                .with_bias(true)
                .with_initializer(Initializer::Zeros)
                .init(device),
        }
    }

    /// Returns the model device without materializing any lazy parameters.
    pub(crate) fn device(&self) -> Device {
        self.coarse_stem.device()
    }

    /// Processes one frame and returns the predictions and advanced state.
    ///
    /// Inputs are `[batch, 64]` and `[batch, 3, 96]`. The batch must be positive,
    /// and every input/state tensor must share the model's physical device.
    /// Shape/device validation takes place before any network computation.
    pub fn forward_frame(
        &self,
        coarse: Tensor<2>,
        fine: Tensor<3>,
        state: ModelState,
    ) -> (ModelOutput, ModelState) {
        let [batch, coarse_bins] = coarse.dims();
        assert!(batch > 0, "model batch must be nonzero");
        assert_eq!(coarse_bins, ERB_BANDS, "coarse input shape");
        assert_eq!(fine.dims(), [batch, 3, DF_BINS], "fine input shape");
        let device = coarse.device();
        assert_eq!(fine.device(), device, "fine input device");
        assert_eq!(self.device(), device, "model input device");
        state.validate(batch, &device);

        let coarse_window = Tensor::cat(
            vec![
                state.coarse_history,
                coarse.reshape([batch, 1, 1, ERB_BANDS]),
            ],
            2,
        );
        let fine_window = Tensor::cat(
            vec![state.fine_history, fine.reshape([batch, 3, 1, DF_BINS])],
            2,
        );
        // Keep *raw* feature windows: the stems' parameters may change between
        // calls, so caching transformed activations would change the model.
        let coarse_history = coarse_window
            .clone()
            .slice([0..batch, 0..1, 1..3, 0..ERB_BANDS]);
        let fine_history = fine_window
            .clone()
            .slice([0..batch, 0..3, 1..5, 0..DF_BINS]);

        let coarse_64 = self.coarse_stem.forward(coarse_window);
        let coarse_32 = self.coarse_down_1.forward(coarse_64.clone());
        let coarse_16 = self.coarse_down_2.forward(coarse_32.clone());
        let fine_96 = self.fine_stem.forward(fine_window);
        let fine_48 = self.fine_down.forward(fine_96.clone());

        // Frequency-major, channel-minor within each branch, coarse first.
        let joint = Tensor::cat(
            vec![
                coarse_16
                    .clone()
                    .permute([0, 2, 3, 1])
                    .reshape([batch, 256]),
                fine_48.clone().permute([0, 2, 3, 1]).reshape([batch, 768]),
            ],
            1,
        );
        let joint = relu(self.joint_projection.forward(joint));
        let [previous_hidden_1, previous_hidden_2] = state.hidden;
        let (mixed_1, hidden_1) = self.recurrent_1.forward(joint, previous_hidden_1);
        let (mixed_2, hidden_2) = self.recurrent_2.forward(mixed_1, previous_hidden_2);
        let decoded = self.decoder_projection.forward(mixed_2);
        let coarse = decoded
            .clone()
            .slice([0..batch, 0..256])
            .reshape([batch, 1, 16, 16])
            .permute([0, 3, 1, 2])
            + coarse_16;
        let fine = decoded
            .slice([0..batch, 256..1024])
            .reshape([batch, 1, 48, 16])
            .permute([0, 3, 1, 2])
            + fine_48;

        // Nearest upsampling, then encoder skip addition, then DW/PW/BN/PReLU.
        let coarse = self.coarse_up_1.forward(nearest_double(coarse) + coarse_32);
        let coarse = self.coarse_up_2.forward(nearest_double(coarse) + coarse_64);
        let fine = self.fine_up.forward(nearest_double(fine) + fine_96);
        let gains = sigmoid(self.gain_head.forward(coarse)).reshape([batch, ERB_BANDS]);
        // Head channel is tap*2 + complex component; frequency is never folded
        // into that axis. This yields [batch, tap, bin, real/imaginary].
        let coefficients = self
            .filter_head
            .forward(fine)
            .tanh()
            .reshape([batch, DF_TAPS, 2, DF_BINS])
            .permute([0, 1, 3, 2]);
        (
            ModelOutput {
                gains,
                coefficients,
            },
            ModelState {
                coarse_history,
                fine_history,
                hidden: [hidden_1, hidden_2],
            },
        )
    }
}

/// Repeat individual frequency positions, not the whole frequency axis.
fn nearest_double(input: Tensor<4>) -> Tensor<4> {
    let [batch, channels, time, frequency] = input.dims();
    input
        .reshape([batch, channels, time, frequency, 1])
        .repeat_dim(4, 2)
        .reshape([batch, channels, time, frequency * 2])
}

#[cfg(all(test, feature = "flex"))]
mod tests {
    use super::*;
    use burn::{module::Param, tensor::TensorData};

    fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
        tensor.into_data().try_to_vec::<f32>().unwrap()
    }

    fn features(frame: usize, device: &Device) -> (Tensor<2>, Tensor<3>) {
        let coarse = (0..2 * ERB_BANDS)
            .map(|index| (index as f32 * 0.13 + frame as f32 * 0.17).sin() * 0.4)
            .collect::<Vec<_>>();
        let fine = (0..2 * 3 * DF_BINS)
            .map(|index| (index as f32 * 0.07 - frame as f32 * 0.11).cos() * 0.3)
            .collect::<Vec<_>>();
        (
            Tensor::from_data(TensorData::new(coarse, [2, ERB_BANDS]), device),
            Tensor::from_data(TensorData::new(fine, [2, 3, DF_BINS]), device),
        )
    }

    #[test]
    fn predictions_preserve_recurrent_history_gradients_until_explicit_detach() {
        let device = Device::flex().autodiff();
        // Fixed BN statistics avoid training-batch normalization coupling this
        // temporal-gradient check to the distribution of the synthetic inputs.
        let model = AnrLite48::new(&device).freeze();
        let (coarse, fine) = features(0, &device);
        let coarse = coarse.require_grad();
        let fine = fine.require_grad();
        let (_, mut state) =
            model.forward_frame(coarse.clone(), fine.clone(), ModelState::new(2, &device));
        let mut late_loss = None;
        for frame in 1..6 {
            let (next_coarse, next_fine) = features(frame, &device);
            let (output, next_state) = model.forward_frame(next_coarse, next_fine, state);
            state = next_state;
            late_loss = Some(output.gains.sum());
        }
        // Frame zero is now outside both raw feature histories; only the
        // recurrent state can carry its influence to the sixth prediction.
        let gradients = late_loss.unwrap().backward();
        for gradient in [
            values(
                coarse
                    .grad(&gradients)
                    .expect("early coarse input must reach late gains"),
            ),
            values(
                fine.grad(&gradients)
                    .expect("early fine input must reach late gains"),
            ),
        ] {
            assert!(gradient.iter().all(|value| value.is_finite()));
            assert!(gradient.iter().any(|value| value.abs() > 0.0));
        }

        let state = state.detach();
        let (fresh_coarse, fresh_fine) = features(6, &device);
        let fresh_coarse = fresh_coarse.require_grad();
        let fresh_fine = fresh_fine.require_grad();
        let (output, _) = model.forward_frame(fresh_coarse.clone(), fresh_fine.clone(), state);
        let gradients = output.gains.sum().backward();
        assert!(
            coarse.grad(&gradients).is_none(),
            "detach must sever early coarse ancestry"
        );
        assert!(
            fine.grad(&gradients).is_none(),
            "detach must sever early fine ancestry"
        );
        for gradient in [
            values(
                fresh_coarse
                    .grad(&gradients)
                    .expect("fresh coarse gradient"),
            ),
            values(fresh_fine.grad(&gradients).expect("fresh fine gradient")),
        ] {
            assert!(gradient.iter().all(|value| value.is_finite()));
            assert!(gradient.iter().any(|value| value.abs() > 0.0));
        }
    }

    #[test]
    fn record_restores_nonzero_filter_head_and_updated_batch_normalization() {
        let training_device = Device::flex().autodiff();
        let inference_device = Device::flex();
        let mut model = AnrLite48::new(&training_device).train();
        model.filter_head.weight = Param::from_tensor(Tensor::full(
            [DF_TAPS * 2, 16, 1, 1],
            0.025,
            &training_device,
        ));
        model.filter_head.bias = Some(Param::from_tensor(Tensor::full(
            [DF_TAPS * 2],
            0.05,
            &training_device,
        )));
        let initial = model.valid();

        // Update actual running BN statistics without an optimizer or backward.
        let (coarse, fine) = features(3, &training_device);
        let _ = model.forward_frame(coarse, fine, ModelState::new(2, &training_device));
        let source = model.valid();
        let restored = AnrLite48::new(&inference_device)
            .load_record(source.clone().into_record())
            .valid();

        let (coarse, fine) = features(0, &inference_device);
        let (before_update, _) = initial.forward_frame(
            coarse.clone(),
            fine.clone(),
            ModelState::new(2, &inference_device),
        );
        let (after_update, _) =
            source.forward_frame(coarse, fine, ModelState::new(2, &inference_device));
        assert!(
            values(before_update.gains)
                .into_iter()
                .zip(values(after_update.gains))
                .any(|(before, after)| (before - after).abs() > 1e-7),
            "training forward must change BN inference behavior"
        );

        let mut source_state = ModelState::new(2, &inference_device);
        let mut restored_state = ModelState::new(2, &inference_device);
        for frame in 0..3 {
            let (coarse, fine) = features(frame, &inference_device);
            let (expected, next_source) =
                source.forward_frame(coarse.clone(), fine.clone(), source_state);
            let (actual, next_restored) = restored.forward_frame(coarse, fine, restored_state);
            let expected_coefficients = values(expected.coefficients);
            assert!(
                expected_coefficients.iter().any(|value| value.abs() > 1e-6),
                "record must exercise a nonzero DF head"
            );
            for (expected, actual) in [
                (values(expected.gains), values(actual.gains)),
                (expected_coefficients, values(actual.coefficients)),
                (
                    values(next_source.hidden[0].clone()),
                    values(next_restored.hidden[0].clone()),
                ),
                (
                    values(next_source.hidden[1].clone()),
                    values(next_restored.hidden[1].clone()),
                ),
            ] {
                for (expected, actual) in expected.into_iter().zip(actual) {
                    assert!(
                        (expected - actual).abs() < 1e-6,
                        "record changed prediction or hidden state: {expected} != {actual}"
                    );
                }
            }
            source_state = next_source;
            restored_state = next_restored;
        }
    }

    #[test]
    fn initialized_filter_head_is_zero_and_gains_are_bounded_for_a_batch() {
        let device = Device::flex();
        let model = AnrLite48::new(&device).valid();
        let (output, state) = model.forward_frame(
            Tensor::full([2, ERB_BANDS], 0.3, &device),
            Tensor::full([2, 3, DF_BINS], -0.2, &device),
            ModelState::new(2, &device),
        );
        assert_eq!(output.coefficients.dims(), [2, DF_TAPS, DF_BINS, 2]);
        assert!(
            values(output.coefficients)
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            values(output.gains)
                .iter()
                .all(|value| *value > 0.0 && *value < 1.0)
        );
        assert!(
            values(state.hidden[0].clone())
                .iter()
                .any(|value| value.abs() > 1e-7)
        );
    }

    #[test]
    fn nearest_upsampling_repeats_positions_instead_of_tiling_the_axis() {
        let device = Device::flex();
        let input = Tensor::from_data([[[[1.0, 2.0, 3.0]]]], &device);
        assert_eq!(
            values(nearest_double(input)),
            [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        );
    }

    #[test]
    #[should_panic]
    fn malformed_state_is_rejected_before_any_layer_runs() {
        let device = Device::flex();
        let model = AnrLite48::new(&device);
        let mut state = ModelState::new(2, &device);
        state.coarse_history = Tensor::zeros([2, 1, 1, ERB_BANDS], &device);
        model.forward_frame(
            Tensor::zeros([2, ERB_BANDS], &device),
            Tensor::zeros([2, 3, DF_BINS], &device),
            state,
        );
    }
}
