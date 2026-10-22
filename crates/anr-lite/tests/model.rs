#![cfg(feature = "flex")]

use anr_lite::{AUDIO_DELAY_SAMPLES, AnrLite48, AudioProcessor, FRAME_SIZE};
use burn::{
    module::{Module, ModuleVisitor, Param},
    tensor::{Device, Gradients, Tensor, TensorData},
};

fn signal(samples: usize, frequency: f32, phase: f32) -> Vec<f32> {
    (0..samples)
        .map(|i| {
            0.12 * (std::f32::consts::TAU * frequency * i as f32 / 48_000.0 + phase).sin()
                + 0.03 * (std::f32::consts::TAU * 2393.0 * i as f32 / 48_000.0).cos()
        })
        .collect()
}

fn waveform(values: Vec<f32>, batch: usize, samples: usize, device: &Device) -> Tensor<2> {
    Tensor::from_data(TensorData::new(values, [batch, samples]), device)
}

#[test]
fn independent_batch_members_match_separate_streams() {
    let device = Device::flex();
    let model = AnrLite48::new(&device).valid();
    let processor = AudioProcessor::new(&device);
    let samples = 2 * FRAME_SIZE + 31;
    let first = signal(samples, 421.0, 0.1);
    let second = signal(samples, 1093.0, 1.2);
    let mut both = first.clone();
    both.extend_from_slice(&second);
    let batched = processor
        .forward(&model, waveform(both, 2, samples, &device))
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    for (stream, source) in [first, second].into_iter().enumerate() {
        let separate = processor
            .forward(&model, waveform(source, 1, samples, &device))
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        for (i, actual) in separate.iter().enumerate() {
            assert!(
                (batched[stream * samples + i] - actual).abs() < 3e-5,
                "batch member {stream}, sample {i} mixed with another stream"
            );
        }
    }
}

#[test]
fn future_audio_cannot_change_output_before_the_declared_context() {
    let device = Device::flex();
    let model = AnrLite48::new(&device).valid();
    let processor = AudioProcessor::new(&device);
    let samples = 8 * FRAME_SIZE;
    let changed_at = 6 * FRAME_SIZE;
    let original = signal(samples, 527.0, 0.3);
    let mut changed = original.clone();
    for value in &mut changed[changed_at..] {
        *value = -*value * 2.0;
    }
    let first = processor
        .forward(&model, waveform(original, 1, samples, &device))
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let second = processor
        .forward(&model, waveform(changed, 1, samples, &device))
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let unchanged = changed_at - AUDIO_DELAY_SAMPLES;
    for i in 0..unchanged {
        assert!(
            (first[i] - second[i]).abs() < 3e-5,
            "future information leaked into output sample {i}"
        );
    }
    let tail_difference = first[changed_at..]
        .iter()
        .zip(&second[changed_at..])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        tail_difference > 1e-4,
        "changed input must affect its corresponding output"
    );
}

struct GradientCheck<'a> {
    gradients: &'a Gradients,
    nonzero_parameters: usize,
}

impl ModuleVisitor for GradientCheck<'_> {
    fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
        if let Some(gradient) = parameter.val().grad(self.gradients) {
            let values = gradient.into_data().try_to_vec::<f32>().unwrap();
            assert!(
                values.iter().all(|v| v.is_finite()),
                "parameter gradient is non-finite"
            );
            if values.iter().any(|v| v.abs() > 1e-12) {
                self.nonzero_parameters += 1;
            }
        }
    }
}

#[test]
fn waveform_loss_reaches_audio_input_and_trainable_parameters() {
    let device = Device::flex().autodiff();
    let model = AnrLite48::new(&device).train();
    let processor = AudioProcessor::new(&device);
    let samples = 2 * FRAME_SIZE;
    let input = waveform(signal(samples, 631.0, 0.4), 1, samples, &device).require_grad();
    let output = processor.forward(&model, input.clone());
    let loss = output.clone().mul(output).mean();
    let gradients = loss.backward();
    let input_gradient = input
        .grad(&gradients)
        .expect("audio input disconnected from waveform loss")
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    assert!(
        input_gradient.iter().all(|v| v.is_finite()),
        "audio gradient is non-finite"
    );
    assert!(
        input_gradient.iter().any(|v| v.abs() > 1e-9),
        "audio gradient is zero"
    );
    let mut check = GradientCheck {
        gradients: &gradients,
        nonzero_parameters: 0,
    };
    model.visit(&mut check);
    assert!(
        check.nonzero_parameters > 0,
        "waveform loss disconnected from model parameters"
    );
}
