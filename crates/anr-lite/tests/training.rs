#![cfg(all(feature = "train", feature = "flex"))]

use std::{fs, path::Path, process::Command};

use anr_lite::{
    AnrLite48, DF_BINS, ERB_BANDS, FRAME_SIZE, ModelState, SAMPLE_RATE,
    burn::{
        module::Module,
        tensor::{Device, Tensor},
    },
};
use hound::{SampleFormat, WavSpec, WavWriter};

fn write_pair(root: &Path, split: &str, id: &str) {
    let samples = 8 * FRAME_SIZE;
    let clean: Vec<i16> = (0..samples)
        .map(|index| {
            let t = index as f32 / SAMPLE_RATE as f32;
            (0.2 * (std::f32::consts::TAU * 500.0 * t).sin() * 32768.0) as i16
        })
        .collect();
    let noisy: Vec<i16> = clean
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let t = index as f32 / SAMPLE_RATE as f32;
            value + (0.025 * (std::f32::consts::TAU * 1200.0 * t).sin() * 32768.0) as i16
        })
        .collect();
    for (kind, data) in [("clean", clean), ("noisy", noisy)] {
        let directory = root.join(format!("{kind}_{split}_wav"));
        fs::create_dir_all(&directory).unwrap();
        let mut writer = WavWriter::create(
            directory.join(format!("{id}.wav")),
            WavSpec {
                channels: 1,
                sample_rate: SAMPLE_RATE,
                bits_per_sample: 16,
                sample_format: SampleFormat::Int,
            },
        )
        .unwrap();
        for sample in data {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
    }
}

#[test]
fn training_persists_an_optimizer_updated_model_and_keeps_evaluation_speakers_separate() {
    let temporary = tempfile::tempdir().unwrap();
    let dataset = temporary.path().join("VoiceBank with spaces");
    write_pair(&dataset, "trainset_28spk", "p001_001");
    write_pair(&dataset, "trainset_28spk", "p001_002");
    write_pair(&dataset, "trainset_28spk", "p002_001");
    write_pair(&dataset, "testset", "p003_001");
    let output = temporary.path().join("training run");
    let result = Command::new(env!("CARGO_BIN_EXE_anr-lite-train"))
        .args(["--dataset"])
        .arg(&dataset)
        .arg("--output-dir")
        .arg(&output)
        .args([
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--segment-seconds",
            "0.04",
            "--validation-speakers",
            "p002",
            "--max-steps",
            "1",
            "--max-validation-items",
            "1",
            "--evaluate-test",
            "--max-test-items",
            "1",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let summary: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("summary.json")).unwrap()).unwrap();
    assert_eq!(summary["steps"], 1);
    assert_eq!(summary["train_items"], 2);
    assert_eq!(summary["validation_items"], 1);
    assert_eq!(summary["test_items"], 1);
    assert_eq!(summary["partial_epoch"], true);
    assert!(
        summary["best_validation_loss"]
            .as_f64()
            .unwrap()
            .is_finite()
    );
    assert!(summary["test_loss"].as_f64().unwrap().is_finite());

    let device = Device::flex();
    let restored = AnrLite48::new(&device)
        .try_load_file(output.join("best.bin"))
        .unwrap()
        .valid();
    let (prediction, _) = restored.forward_frame(
        Tensor::ones([1, ERB_BANDS], &device),
        Tensor::ones([1, 3, DF_BINS], &device),
        ModelState::new(1, &device),
    );
    let coefficients = prediction.coefficients.try_into_vec_as::<f32>().unwrap();
    assert!(coefficients.iter().all(|value| value.is_finite()));
    // BatchNorm updates cannot change a zero-initialized residual head. Nonzero
    // coefficients after reloading therefore prove an optimizer update persisted.
    assert!(
        coefficients.iter().any(|value| value.abs() > 1e-8),
        "checkpoint contains no learned residual-filter coefficients"
    );
}
