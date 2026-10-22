#![cfg(all(feature = "train", feature = "flex"))]

use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

use anr_lite::{
    AnrLite48, FRAME_SIZE, SAMPLE_RATE,
    burn::{
        module::{Module, ModuleMapper, Param},
        tensor::{Device, Tensor},
    },
};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

struct ZeroParameters;

impl ModuleMapper for ZeroParameters {
    fn map_float<const D: usize>(&mut self, parameter: Param<Tensor<D>>) -> Param<Tensor<D>> {
        parameter.map(|tensor| Tensor::zeros(tensor.dims(), &tensor.device()))
    }
}

fn checkpoint(path: &Path) {
    AnrLite48::new(&Device::flex())
        .map(&mut ZeroParameters)
        .valid()
        .save_file(path)
        .unwrap();
}

fn pcm16(path: &Path, samples: &[i16], rate: u32, channels: u16) {
    let mut writer = WavWriter::create(
        path,
        WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        },
    )
    .unwrap();
    for &sample in samples {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
}

fn render(checkpoint: &Path, input: &Path, output: &Path, extra: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_anr-lite-evaluate"));
    command
        .arg("--checkpoint")
        .arg(checkpoint)
        .arg("--input-dir")
        .arg(input)
        .arg("--output-dir")
        .arg(output);
    if !extra.contains(&"--device") {
        command.args(["--device", "cpu"]);
    }
    if !extra.contains(&"--batch-size") {
        command.args(["--batch-size", "2"]);
    }
    command.args(extra).output().unwrap()
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_failure(output: &Output) {
    assert!(
        !output.status.success(),
        "unexpected success: stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn real_record_renders_unequal_pcm16_signals_with_known_half_gain_and_exact_lengths() {
    let temporary = tempfile::tempdir().unwrap();
    let record = temporary.path().join("zero model.bin");
    checkpoint(&record);
    let input = temporary.path().join("noisy inputs");
    fs::create_dir(&input).unwrap();
    // Distinct amplitudes, DC, impulses and non-hop tails catch stream mixing,
    // normalization, integer scaling, latency and padded-batch trimming errors.
    let mut alpha: Vec<i16> = (0..FRAME_SIZE + 37)
        .map(|index| ((index * 197 % 17001) as i32 - 6000) as i16)
        .collect();
    alpha[0] = i16::MIN;
    alpha[FRAME_SIZE + 36] = i16::MAX;
    let mut zulu: Vec<i16> = (0..2 * FRAME_SIZE + 113)
        .map(|index| ((index * 79 % 5003) as i32 - 3200) as i16)
        .collect();
    zulu[319] = 27000;
    zulu[2 * FRAME_SIZE + 112] = -19000;
    // Reverse creation order must not become the manifest/evaluation order.
    pcm16(&input.join("zulu.wav"), &zulu, SAMPLE_RATE, 1);
    pcm16(&input.join("alpha.wav"), &alpha, SAMPLE_RATE, 1);
    fs::write(input.join("not-a-wave.txt"), "ignored non-WAV").unwrap();

    let output = temporary.path().join("rendered");
    assert_success(&render(&record, &input, &output, &[]));
    for (name, source) in [("alpha.wav", &alpha), ("zulu.wav", &zulu)] {
        let mut reader = WavReader::open(output.join(name)).unwrap();
        assert_eq!(
            reader.spec(),
            WavSpec {
                channels: 1,
                sample_rate: SAMPLE_RATE,
                bits_per_sample: 32,
                sample_format: SampleFormat::Float,
            }
        );
        let actual: Vec<f32> = reader.samples::<f32>().collect::<Result<_, _>>().unwrap();
        assert_eq!(actual.len(), source.len(), "length of {name}");
        for (index, (&sample, &value)) in source.iter().zip(&actual).enumerate() {
            let expected = f32::from(sample) / 65536.0;
            assert!(
                (value - expected).abs() < 3e-5,
                "{name} sample {index}: expected {expected}, got {value}"
            );
        }
        // Inputs remain integer PCM with exactly the original sample values.
        let unchanged: Vec<i16> = WavReader::open(input.join(name))
            .unwrap()
            .samples::<i16>()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(&unchanged, source);
    }
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["source_count"], 2);
    assert_eq!(manifest["evaluated_count"], 2);
    assert_eq!(manifest["files"][0]["filename"], "alpha.wav");
    assert_eq!(manifest["files"][0]["input_samples"], alpha.len());
    assert_eq!(manifest["files"][0]["output_samples"], alpha.len());
    assert_eq!(manifest["files"][1]["filename"], "zulu.wav");
    assert_eq!(manifest["files"][1]["output_samples"], zulu.len());

    let limited = temporary.path().join("limited");
    assert_success(&render(&record, &input, &limited, &["--limit", "1"]));
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(limited.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["source_count"], 2);
    assert_eq!(manifest["evaluated_count"], 1);
    assert_eq!(manifest["limit"], 1);
    assert_eq!(manifest["files"].as_array().unwrap().len(), 1);
    assert!(limited.join("alpha.wav").is_file());
    assert!(!limited.join("zulu.wav").exists());

    assert_failure(&render(&record, &input, &output, &[]));
    assert_failure(&render(
        &record,
        &input,
        &temporary.path().join("bad-limit"),
        &["--limit", "0"],
    ));
    assert_failure(&render(
        &record,
        &input,
        &temporary.path().join("bad-batch"),
        &["--batch-size", "0"],
    ));
    assert_failure(&render(
        &temporary.path().join("missing.bin"),
        &input,
        &temporary.path().join("missing-checkpoint"),
        &[],
    ));
    #[cfg(not(feature = "wgpu"))]
    assert_failure(&render(
        &record,
        &input,
        &temporary.path().join("unavailable-backend"),
        &["--device", "wgpu"],
    ));
}

#[test]
fn rendering_rejects_non_pcm16_mono_48k_and_truncated_or_empty_waves() {
    let temporary = tempfile::tempdir().unwrap();
    let record = temporary.path().join("zero.bin");
    checkpoint(&record);
    for (case, rate, channels, samples) in [
        ("wrong-rate", 16000, 1, vec![5, 10]),
        ("stereo", SAMPLE_RATE, 2, vec![5, 10]),
        ("empty", SAMPLE_RATE, 1, vec![]),
        ("truncated", SAMPLE_RATE, 1, vec![5, 10]),
    ] {
        let input = temporary.path().join(case);
        fs::create_dir(&input).unwrap();
        let path = input.join("bad.wav");
        pcm16(&path, &samples, rate, channels);
        if case == "truncated" {
            let length = fs::metadata(&path).unwrap().len();
            fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap()
                .set_len(length - 2)
                .unwrap();
        }
        assert_failure(&render(
            &record,
            &input,
            &temporary.path().join(format!("{case}-out")),
            &[],
        ));
    }
    let input = temporary.path().join("float");
    fs::create_dir(&input).unwrap();
    let mut writer = WavWriter::create(
        input.join("float.wav"),
        WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        },
    )
    .unwrap();
    writer.write_sample(0.25f32).unwrap();
    writer.finalize().unwrap();
    assert_failure(&render(
        &record,
        &input,
        &temporary.path().join("float-out"),
        &[],
    ));
}
