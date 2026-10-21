use anr_core::{Error, NoiseReduction};
use anr_ort::{DFN3, ort::session::Session};
use std::sync::LazyLock;

static ENVIRONMENT: LazyLock<anr_ort::ort::environment::Environment> = LazyLock::new(|| {
    anr_ort::initialize_runtime().unwrap();
    anr_ort::ort::init().build().unwrap()
});

fn environment() -> &'static anr_ort::ort::environment::Environment {
    &ENVIRONMENT
}

// A small ONNX test graph, not a DeepFilterNet model:
// audio = input + mean(input) + mean(states); new_states = states + atten_lim_db.
// Its deliberately reordered inputs/outputs detect positional binding mistakes.
fn model() -> DFN3 {
    let session = Session::builder(environment())
        .unwrap()
        .with_intra_threads(1)
        .unwrap()
        .commit_from_memory(include_bytes!("fixtures/stateful.onnx"))
        .unwrap();
    let mut model = DFN3::new(session).unwrap();
    model.set_factor(10.0); // 6 dB, hence a state increment of 6 in this test graph.
    model
}

#[test]
fn scalar_attenuation_and_symbolic_audio_shape_are_supported() {
    let session = Session::builder(environment())
        .unwrap()
        .with_intra_threads(1)
        .unwrap()
        .commit_from_memory(include_bytes!("fixtures/scalar-symbolic.onnx"))
        .unwrap();
    let mut model = DFN3::new(session).unwrap();
    model.set_factor(10.0);
    let mut output = [0.0; 480];
    model.run(48_000, &[1.0; 480], &mut output).unwrap();
    model.run(48_000, &[1.0; 480], &mut output).unwrap();
    assert_eq!(output, [8.0; 480]);
}

#[test]
fn state_survives_calls_and_reset_starts_a_new_stream() {
    let mut model = model();
    let input = [1.0; 480];
    let mut output = [0.0; 480];
    model.run(48_000, &input, &mut output).unwrap();
    assert_eq!(output, [2.0; 480]);
    model.run(48_000, &input, &mut output).unwrap();
    assert_eq!(output, [8.0; 480]);
    model.reset();
    model.run(48_000, &input, &mut output).unwrap();
    assert_eq!(output, [2.0; 480]);
    model.run(48_000, &input, &mut output).unwrap();
    assert_eq!(output, [8.0; 480]); // reset must preserve strength.
}

#[test]
fn partial_frames_are_zero_padded_without_leaking_previous_input() {
    let mut model = model();
    let mut input = [1.0; 481];
    input[480] = 2.0;
    let mut output = [0.0; 481];
    model.run(48_000, &input, &mut output).unwrap();
    assert_eq!(&output[..480], &[2.0; 480]);
    assert!((output[480] - (8.0 + 2.0 / 480.0)).abs() < 1e-5);
    let mut next = [0.0];
    model.run(48_000, &[0.0], &mut next).unwrap();
    assert_eq!(next, [12.0]);
}

#[test]
fn aligned_chunking_matches_one_call() {
    let input = vec![0.25; 480 * 3];
    let mut whole = vec![0.0; input.len()];
    model().run(48_000, &input, &mut whole).unwrap();
    let mut split = vec![0.0; input.len()];
    let mut model = model();
    for (input, output) in input.chunks(480).zip(split.chunks_mut(480)) {
        model.run(48_000, input, output).unwrap();
    }
    assert_eq!(whole, split);
    assert_eq!(split[960], 12.5);
}

#[test]
fn invalid_buffers_do_not_write_output_or_advance_state() {
    let mut model = model();
    let mut output = [99.0; 480];
    assert!(matches!(
        model.run(16_000, &[1.0; 480], &mut output),
        Err(Error::InvalidSampleRate {
            expected: 48_000,
            actual: 16_000
        })
    ));
    assert_eq!(output, [99.0; 480]);
    assert!(matches!(
        model.run(48_000, &[], &mut output),
        Err(Error::EmptyInput)
    ));
    assert_eq!(output, [99.0; 480]);
    for length in [479, 481] {
        let mut wrong = vec![99.0; length];
        assert!(matches!(
            model.run(48_000, &[1.0; 480], &mut wrong),
            Err(Error::OutputLengthMismatch { input: 480, output }) if output == length
        ));
        assert!(wrong.iter().all(|&sample| sample == 99.0));
    }
    model.run(48_000, &[1.0; 480], &mut output).unwrap();
    assert_eq!(output, [2.0; 480]);
}

#[test]
fn strength_is_bounded_and_nan_does_not_poison_inference() {
    let mut model = model();
    model.set_factor(f32::NAN);
    model.set_atten_lim_db(f32::NAN);
    let mut output = [0.0; 480];
    model.run(48_000, &[1.0; 480], &mut output).unwrap();
    model.run(48_000, &[1.0; 480], &mut output).unwrap();
    assert_eq!(output, [8.0; 480]);
    model.reset();
    model.set_factor(f32::NEG_INFINITY);
    model.run(48_000, &[1.0; 480], &mut output).unwrap();
    model.run(48_000, &[1.0; 480], &mut output).unwrap();
    assert_eq!(output, [2.0; 480]);
    model.reset();
    model.set_atten_lim_db(f32::INFINITY);
    model.run(48_000, &[1.0; 480], &mut output).unwrap();
    model.run(48_000, &[1.0; 480], &mut output).unwrap();
    assert_eq!(output, [62.0; 480]);
}

#[test]
fn rejects_an_incompatible_model_before_processing_audio() {
    let session = Session::builder(environment())
        .unwrap()
        .with_intra_threads(1)
        .unwrap()
        .commit_from_memory(include_bytes!("fixtures/wrong-state-size.onnx"))
        .unwrap();
    assert!(matches!(DFN3::new(session), Err(Error::InvalidModel(_))));
}

#[test]
#[ignore = "requires a user-supplied DFN3 model; set ANR_DFN3_MODEL to its path"]
fn supplied_dfn3_model_produces_finite_audio_and_reset_is_repeatable() {
    let path = std::env::var_os("ANR_DFN3_MODEL").expect("set ANR_DFN3_MODEL");
    let mut model = DFN3::from_file(environment(), path).unwrap();
    model.set_factor(80.0);
    let input: Vec<f32> = (0..4_800)
        .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.1)
        .collect();
    let mut first = vec![0.0; input.len()];
    model.run(48_000, &input, &mut first).unwrap();
    assert!(first.iter().all(|x| x.is_finite()));
    assert!(first.iter().any(|x| x.abs() > 1e-7));
    model.reset();
    let mut second = vec![0.0; input.len()];
    model.run(48_000, &input, &mut second).unwrap();
    assert!(first.iter().zip(second).all(|(a, b)| (a - b).abs() < 1e-5));
}
