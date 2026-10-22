#![cfg(feature = "flex")]

use anr_lite::{AUDIO_DELAY_SAMPLES, AnrLite48, AudioProcessor, FRAME_SIZE};
use burn::{
    module::Module,
    tensor::{Device, Tensor, TensorData},
};

fn signal(samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|i| {
            let t = i as f32 / 48_000.0;
            0.15 * (std::f32::consts::TAU * 317.0 * t).sin()
                + 0.04 * (std::f32::consts::TAU * 1259.0 * t).cos()
        })
        .collect()
}

#[test]
fn streaming_flush_matches_aligned_waveform_with_partial_last_frame() {
    let device = Device::flex();
    device.seed(27);
    let model = AnrLite48::new(&device).valid();
    let processor = AudioProcessor::new(&device);
    let samples = FRAME_SIZE * 5 + 137;
    let input = signal(samples);
    let whole = processor
        .forward(
            &model,
            Tensor::<2>::from_data(TensorData::new(input.clone(), [1, samples]), &device),
        )
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let mut state = processor.initial_state(1);
    let mut streamed = Vec::new();
    for chunk in input.chunks(FRAME_SIZE) {
        let mut frame = vec![0.0; FRAME_SIZE];
        frame[..chunk.len()].copy_from_slice(chunk);
        let output = processor.forward_frame(
            &model,
            Tensor::<2>::from_data(TensorData::new(frame, [1, FRAME_SIZE]), &device),
            &mut state,
        );
        streamed.extend(output.into_data().try_to_vec::<f32>().unwrap());
    }
    streamed.extend(
        processor
            .flush(&model, &mut state)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap(),
    );
    let aligned = &streamed[AUDIO_DELAY_SAMPLES..AUDIO_DELAY_SAMPLES + samples];
    for (i, (&a, &b)) in whole.iter().zip(aligned).enumerate() {
        assert!((a - b).abs() < 3e-5, "sample {i}: whole={a}, stream={b}");
    }
    let energy: f32 = whole.iter().map(|v| v * v).sum();
    assert!(
        energy > 0.01,
        "nonzero speech must not be discarded by frame alignment"
    );
}

#[test]
fn replacing_state_after_a_nonzero_prefix_starts_an_independent_stream() {
    let device = Device::flex();
    let model = AnrLite48::new(&device).valid();
    let processor = AudioProcessor::new(&device);
    let fresh_processor = AudioProcessor::new(&device);
    let mut reused = processor.initial_state(1);
    for _ in 0..5 {
        let _ = processor.forward_frame(
            &model,
            Tensor::full([1, FRAME_SIZE], 0.3, &device),
            &mut reused,
        );
    }
    reused = processor.initial_state(1);
    let mut fresh = fresh_processor.initial_state(1);
    for chunk in signal(5 * FRAME_SIZE).chunks(FRAME_SIZE) {
        let input =
            Tensor::<2>::from_data(TensorData::new(chunk.to_vec(), [1, FRAME_SIZE]), &device);
        let actual = processor
            .forward_frame(&model, input.clone(), &mut reused)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let expected = fresh_processor
            .forward_frame(&model, input, &mut fresh)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 3e-5, "reset retained previous stream state");
        }
    }
}
