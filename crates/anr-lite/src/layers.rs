//! Device-native grouped projections, recurrent cells and convolution blocks.

use burn::module::{Module, Param};
use burn::nn::{
    BatchNorm, BatchNormConfig, Initializer, Linear, LinearConfig, PRelu, PReluConfig,
    PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
};
use burn::tensor::{Device, Tensor, activation::sigmoid};

/// A grouped affine transform using one batched matrix multiplication.
///
/// Parameters are `[group, input_per_group, output_per_group]`; the contiguous
/// feature ranges belonging to different groups never interact.
#[derive(Module, Debug)]
pub(crate) struct GroupedLinear {
    weight: Param<Tensor<3>>,
    bias: Param<Tensor<2>>,
}

impl GroupedLinear {
    pub(crate) fn new(input: usize, output: usize, groups: usize, device: &Device) -> Self {
        assert!(
            input > 0 && output > 0 && groups > 0,
            "grouped affine dimensions must be nonzero"
        );
        assert_eq!(input % groups, 0, "input must be divisible by groups");
        assert_eq!(output % groups, 0, "output must be divisible by groups");
        let input_width = input / groups;
        let output_width = output / groups;
        let bound = 1.0 / (input_width as f64).sqrt();
        let initializer = Initializer::Uniform {
            min: -bound,
            max: bound,
        };
        Self {
            weight: initializer.init([groups, input_width, output_width], device),
            bias: initializer.init([groups, output_width], device),
        }
    }

    pub(crate) fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        let [batch, features] = input.dims();
        let [groups, input_width, output_width] = self.weight.lazy_shape().dims();
        assert!(batch > 0, "grouped affine batch must be nonzero");
        assert_eq!(features, groups * input_width, "grouped affine input shape");
        assert_eq!(
            self.bias.lazy_shape().dims::<2>(),
            [groups, output_width],
            "grouped affine bias shape"
        );
        assert_eq!(
            input.device(),
            self.weight.lazy_device(),
            "grouped affine input device"
        );
        assert_eq!(
            input.device(),
            self.bias.lazy_device(),
            "grouped affine bias device"
        );
        let grouped = input
            .reshape([batch, groups, input_width])
            .permute([1, 0, 2]);
        let output = grouped.matmul(self.weight.val()) + self.bias.val().unsqueeze_dim::<3>(1);
        output
            .permute([1, 0, 2])
            .reshape([batch, groups * output_width])
    }
}

/// Grouped GRU using ONNX `linear_before_reset=1` / PyTorch equations.
///
/// Each group's gate layout is `[z, r, n]`, with separate input and hidden
/// biases. In particular, the candidate hidden bias is *inside* the reset:
/// `n = tanh(Wn*x + bin + r * (Rn*h + bhn))`.
#[derive(Module, Debug)]
pub(crate) struct GroupedGruCell {
    input: GroupedLinear,
    hidden: GroupedLinear,
}

impl GroupedGruCell {
    pub(crate) fn new(width: usize, groups: usize, device: &Device) -> Self {
        assert!(width > 0 && groups > 0, "GRU dimensions must be nonzero");
        assert_eq!(width % groups, 0, "GRU width must be divisible by groups");
        Self {
            input: GroupedLinear::new(width, 3 * width, groups, device),
            hidden: GroupedLinear::new(width, 3 * width, groups, device),
        }
    }

    pub(crate) fn forward(&self, input: Tensor<2>, hidden: Tensor<2>) -> Tensor<2> {
        let [batch, width] = input.dims();
        let [groups, group_width, _] = self.input.weight.lazy_shape().dims();
        assert!(batch > 0, "GRU batch must be nonzero");
        assert_eq!(width, groups * group_width, "GRU input shape");
        assert_eq!(hidden.dims(), [batch, width], "GRU hidden shape");
        assert_eq!(input.device(), hidden.device(), "GRU hidden device");
        assert_eq!(
            input.device(),
            self.input.weight.lazy_device(),
            "GRU input device"
        );
        assert_eq!(
            input.device(),
            self.hidden.weight.lazy_device(),
            "GRU parameter device"
        );
        let input_gates = self
            .input
            .forward(input)
            .reshape([batch, groups, 3, group_width]);
        let hidden_gates =
            self.hidden
                .forward(hidden.clone())
                .reshape([batch, groups, 3, group_width]);
        let gate = |tensor: &Tensor<4>, index| {
            tensor
                .clone()
                .slice([0..batch, 0..groups, index..index + 1, 0..group_width])
                .reshape([batch, width])
        };
        let update = sigmoid(gate(&input_gates, 0) + gate(&hidden_gates, 0));
        let reset = sigmoid(gate(&input_gates, 1) + gate(&hidden_gates, 1));
        let candidate = (gate(&input_gates, 2) + reset * gate(&hidden_gates, 2)).tanh();
        update.clone() * hidden + (update.neg() + 1.0) * candidate
    }
}

/// The mixed residual activation is not the state of the GRU cell.
#[derive(Module, Debug)]
pub(crate) struct RecurrentBlock {
    cell: GroupedGruCell,
    mixing: Linear,
}

impl RecurrentBlock {
    pub(crate) fn new(device: &Device) -> Self {
        Self {
            cell: GroupedGruCell::new(128, 4, device),
            mixing: LinearConfig::new(128, 128).init(device),
        }
    }

    pub(crate) fn forward(&self, input: Tensor<2>, hidden: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
        let next_hidden = self.cell.forward(input.clone(), hidden);
        let mixed = self.mixing.forward(next_hidden.clone()) + input;
        (mixed, next_hidden)
    }
}

#[derive(Module, Debug)]
pub(crate) struct Stem {
    conv: Conv2d,
    norm: BatchNorm,
    activation: PRelu,
}

impl Stem {
    pub(crate) fn new(channels: usize, time_kernel: usize, device: &Device) -> Self {
        Self {
            conv: Conv2dConfig::new([channels, 16], [time_kernel, 3])
                .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
                .with_bias(false)
                .init(device),
            norm: BatchNormConfig::new(16).init(device),
            activation: PReluConfig::new().with_num_parameters(16).init(device),
        }
    }

    pub(crate) fn device(&self) -> Device {
        self.conv.weight.lazy_device()
    }

    pub(crate) fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        self.activation
            .forward(self.norm.forward(self.conv.forward(input)))
    }
}

#[derive(Module, Debug)]
pub(crate) struct SeparableBlock {
    depthwise: Conv2d,
    pointwise: Conv2d,
    norm: BatchNorm,
    activation: PRelu,
}

impl SeparableBlock {
    pub(crate) fn new(frequency_stride: usize, device: &Device) -> Self {
        Self {
            depthwise: Conv2dConfig::new([16, 16], [1, 5])
                .with_groups(16)
                .with_stride([1, frequency_stride])
                .with_padding(PaddingConfig2d::Explicit(0, 2, 0, 2))
                .with_bias(false)
                .init(device),
            pointwise: Conv2dConfig::new([16, 16], [1, 1])
                .with_bias(false)
                .init(device),
            norm: BatchNormConfig::new(16).init(device),
            activation: PReluConfig::new().with_num_parameters(16).init(device),
        }
    }

    pub(crate) fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let output = self.pointwise.forward(self.depthwise.forward(input));
        self.activation.forward(self.norm.forward(output))
    }
}

#[cfg(all(test, feature = "flex"))]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
        tensor.into_data().try_to_vec::<f32>().unwrap()
    }

    #[test]
    fn grouped_affine_keeps_batches_and_groups_independent() {
        let device = Device::flex();
        let mut layer = GroupedLinear::new(4, 4, 2, &device);
        layer.weight = Param::from_tensor(Tensor::from_data(
            [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 3.0]]],
            &device,
        ));
        layer.bias = Param::from_tensor(Tensor::zeros([2, 2], &device));
        let input = Tensor::from_data([[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 4.0]], &device);
        assert_eq!(
            values(layer.forward(input)),
            [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 6.0, 12.0]
        );
    }

    #[test]
    fn gru_resets_after_hidden_affine_and_preserves_nonzero_recurrence() {
        let device = Device::flex();
        let mut cell = GroupedGruCell::new(4, 2, &device);
        // Per-group gate order is z, r, n. r=0.25, z=0.5. The hidden
        // candidate bias must be reset as well (linear_before_reset=1).
        cell.input.weight = Param::from_tensor(Tensor::zeros([2, 2, 6], &device));
        cell.hidden.weight = Param::from_tensor(Tensor::zeros([2, 2, 6], &device));
        let input_bias = [0.0, 0.0, -(3.0_f32).ln(), -(3.0_f32).ln(), 0.2, 0.2];
        let hidden_bias = [0.0, 0.0, 0.0, 0.0, 0.8, 0.8];
        cell.input.bias = Param::from_tensor(Tensor::from_data(
            TensorData::new(input_bias.repeat(2), [2, 6]),
            &device,
        ));
        cell.hidden.bias = Param::from_tensor(Tensor::from_data(
            TensorData::new(hidden_bias.repeat(2), [2, 6]),
            &device,
        ));
        let initial = Tensor::full([2, 4], 0.6, &device);
        let first = cell.forward(Tensor::zeros([2, 4], &device), initial);
        let second = cell.forward(Tensor::zeros([2, 4], &device), first);
        let candidate = 0.4_f32.tanh();
        let expected = 0.5 * (0.5 * 0.6 + 0.5 * candidate) + 0.5 * candidate;
        for value in values(second) {
            assert!((value - expected).abs() < 1e-6, "{value} != {expected}");
        }
    }

    #[test]
    fn recurrent_hidden_preserves_gradient_into_previous_frame() {
        let device = Device::flex().autodiff();
        let mut cell = GroupedGruCell::new(4, 2, &device);
        cell.input.weight = Param::from_tensor(Tensor::full([2, 2, 6], 0.1, &device));
        cell.hidden.weight = Param::from_tensor(Tensor::full([2, 2, 6], 0.1, &device));
        cell.input.bias = Param::from_tensor(Tensor::zeros([2, 6], &device));
        cell.hidden.bias = Param::from_tensor(Tensor::zeros([2, 6], &device));
        let previous = Tensor::full([2, 4], 0.3, &device).require_grad();
        let first = cell.forward(previous.clone(), Tensor::zeros([2, 4], &device));
        let output = cell.forward(Tensor::full([2, 4], 0.2, &device), first);
        let gradients = output.sum().backward();
        let gradient = previous
            .grad(&gradients)
            .expect("previous frame must remain connected");
        assert!(
            values(gradient)
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
        );
    }
}
