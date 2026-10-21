use anr_core::{Error, NoiseReduction};
use anr_wgpu::DFN3;

#[test]
fn rejects_unrelated_model_before_opening_a_graphics_device() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unrelated.onnx");
    std::fs::write(&path, b"not the supported DFN3 export").unwrap();
    assert!(matches!(DFN3::from_file(path), Err(Error::InvalidModel(_))));
}

fn samples(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn compare_reference(model: &mut DFN3) {
    let input = samples(include_bytes!("fixtures/reference_input.f32le"));
    let expected = samples(include_bytes!("fixtures/reference_output.f32le"));
    let mut actual = vec![f32::NAN; input.len()];
    // Whole-frame calls may be batched differently; only the final 13 samples
    // are padded. Strength changes deliberately exercise both attenuation branches.
    let sections = [
        (0, 12 * 480, 80.0),
        (12 * 480, 24 * 480, 0.0),
        (24 * 480, 36 * 480, 5.0),
        (36 * 480, 60 * 480 + 13, 100.0),
    ];
    for (start, end, strength) in sections {
        model.set_factor(strength);
        let mut cursor = start;
        while cursor < end {
            let next = (cursor + 5 * 480).min(end);
            model
                .run(48_000, &input[cursor..next], &mut actual[cursor..next])
                .unwrap();
            cursor = next;
        }
    }
    let mut maximum = 0.0_f32;
    let mut squared = 0.0_f64;
    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        assert!(got.is_finite(), "non-finite sample {index}");
        let error = (got - want).abs();
        maximum = maximum.max(error);
        squared += f64::from(error).powi(2);
        assert!(
            error < 2.0e-5,
            "sample {index}: got {got}, expected {want}, error {error}"
        );
    }
    let rms = (squared / actual.len() as f64).sqrt();
    assert!(rms < 2.0e-6, "maximum error {maximum}, RMS error {rms}");
    eprintln!("DFN3 wgpu reference: maximum={maximum:e}, RMS={rms:e}");
}

#[test]
#[ignore = "requires ANR_DFN3_MODEL and a native hardware wgpu adapter"]
fn supplied_model_matches_streaming_reference_after_invalid_calls_thread_move_and_reset() {
    let path = std::env::var_os("ANR_DFN3_MODEL").expect("set ANR_DFN3_MODEL");
    let mut model = DFN3::from_file(path).unwrap();
    let mut untouched = [123.0; 480];
    assert!(matches!(
        model.run(44_100, &[0.5; 480], &mut untouched),
        Err(Error::InvalidSampleRate { .. })
    ));
    assert_eq!(untouched, [123.0; 480]);
    assert!(matches!(
        model.run(48_000, &[0.5; 479], &mut untouched),
        Err(Error::OutputLengthMismatch { .. })
    ));
    assert_eq!(untouched, [123.0; 480]);
    assert!(matches!(
        model.run(48_000, &[], &mut []),
        Err(Error::EmptyInput)
    ));
    let mut model = std::thread::spawn(move || {
        compare_reference(&mut model);
        model
    })
    .join()
    .unwrap();
    model.set_factor(37.0);
    let strength = model.factor();
    model.reset();
    assert_eq!(model.factor(), strength);
    compare_reference(&mut model);
}
