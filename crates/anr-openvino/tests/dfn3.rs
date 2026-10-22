use anr_core::{Error, NoiseReduction};
use anr_openvino::{DFN3, Device};

#[test]
fn rejects_unrelated_model_before_opening_runtime_or_device() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unrelated.onnx");
    std::fs::write(&path, b"not the supported DFN3 export").unwrap();
    for device in [Device::Cpu, Device::Gpu(u32::MAX), Device::Npu] {
        assert!(matches!(
            DFN3::from_file_with_device(&path, device),
            Err(Error::InvalidModel(_))
        ));
    }
    assert!(matches!(DFN3::from_file(path), Err(Error::InvalidModel(_))));
}

#[test]
fn explicit_devices_parse_without_accepting_fallback_plugins() {
    for (text, want) in [
        ("CPU", Device::Cpu),
        ("cpu", Device::Cpu),
        ("GPU", Device::Gpu(0)),
        ("gpu.0", Device::Gpu(0)),
        ("GPU.12", Device::Gpu(12)),
        ("NPU", Device::Npu),
        ("npu", Device::Npu),
    ] {
        assert_eq!(text.parse::<Device>().unwrap(), want);
        assert_eq!(want.to_string().parse::<Device>().unwrap(), want);
    }
    for text in [
        "",
        "AUTO",
        "AUTO:GPU,CPU",
        "HETERO:GPU,CPU",
        "CPU.0",
        "NPU.0",
        "GPU.",
        "GPU.-1",
        "GPU.+1",
        "GPU.4294967296",
        "GPU.1.0",
        " GPU",
        "GPU ",
    ] {
        assert!(text.parse::<Device>().is_err(), "accepted {text:?}");
    }
    assert_eq!(Device::default(), Device::Cpu);
}

fn samples(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn invalid_calls_preserve_output(model: &mut DFN3) {
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
}

fn compare_reference(model: &mut DFN3) {
    let input = samples(include_bytes!("fixtures/reference_input.f32le"));
    let expected = samples(include_bytes!("fixtures/reference_output.f32le"));
    assert_eq!(input.len(), expected.len());
    let mut actual = vec![f32::NAN; input.len()];
    for (start, end, strength) in [
        (0, 12 * 480, 80.0),
        (12 * 480, 24 * 480, 0.0),
        (24 * 480, 36 * 480, 5.0),
        (36 * 480, 60 * 480 + 13, 100.0),
    ] {
        // This also catches validation accidentally resetting a live stream.
        invalid_calls_preserve_output(model);
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
    eprintln!("DFN3 OpenVINO reference: maximum={maximum:e}, RMS={rms:e}");
}

fn streaming_regression(device: Device) {
    let path = std::env::var_os("ANR_DFN3_MODEL").expect("set ANR_DFN3_MODEL");
    let mut model = DFN3::from_file_with_device(&path, device).unwrap();
    invalid_calls_preserve_output(&mut model);
    let mut model = std::thread::spawn(move || {
        compare_reference(&mut model);
        model
    })
    .join()
    .unwrap();
    model.set_factor(37.0);
    let strength = model.factor();
    model.set_factor(f32::NAN);
    assert_eq!(model.factor(), strength);
    model.reset();
    assert_eq!(model.factor(), strength);
    compare_reference(&mut model);
    // Reset must retain the actual attenuation input, not only the getter.
    let input = samples(include_bytes!("fixtures/reference_input.f32le"));
    let input = &input[..12 * 480];
    let mut first = vec![0.0; input.len()];
    let mut second = vec![0.0; input.len()];
    let mut fresh = DFN3::from_file_with_device(&path, device).unwrap();
    fresh.set_factor(37.0);
    fresh.run(48_000, input, &mut first).unwrap();
    model.set_factor(37.0);
    model.reset();
    model.run(48_000, input, &mut second).unwrap();
    assert_eq!(first, second);
    model.set_factor(f32::INFINITY);
    assert_eq!(model.factor(), 100.0);
    model.set_factor(f32::NEG_INFINITY);
    assert_eq!(model.factor(), 0.0);
}

#[test]
#[ignore = "requires ANR_DFN3_MODEL and the OpenVINO CPU plugin"]
fn cpu_matches_streaming_reference_invalid_calls_thread_move_and_reset() {
    streaming_regression(Device::Cpu);
}

#[test]
#[ignore = "requires ANR_DFN3_MODEL and a supported hardware OpenVINO GPU"]
fn gpu_matches_streaming_reference_invalid_calls_thread_move_and_reset() {
    let device = std::env::var("ANR_OPENVINO_DEVICE")
        .unwrap_or_else(|_| "GPU".into())
        .parse()
        .unwrap();
    assert!(matches!(device, Device::Gpu(_)));
    streaming_regression(device);
}

#[test]
#[ignore = "requires ANR_DFN3_MODEL and an installed OpenVINO Runtime"]
fn unavailable_explicit_gpu_returns_an_error_instead_of_cpu_fallback() {
    let path = std::env::var_os("ANR_DFN3_MODEL").expect("set ANR_DFN3_MODEL");
    assert!(DFN3::from_file_with_device(path, Device::Gpu(u32::MAX)).is_err());
}
