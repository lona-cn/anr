#![cfg(all(
    target_os = "windows",
    feature = "ort-cuda",
    feature = "ort-tensorrt",
    feature = "ort-directml"
))]

use std::{path::PathBuf, process::Command};

#[test]
#[ignore = "requires a Windows GPU and matching ORT CUDA, TensorRT, and DirectML libraries"]
fn gpu_providers_share_environment_and_complete_streaming_measurements() {
    // The existing self-authored graph exercises GPU execution without claiming
    // compatibility or performance for the real DFN3 weights.
    let model =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../anr-cli/tests/fixtures/control.onnx");
    let output = Command::new(env!("CARGO_BIN_EXE_anr-bench"))
        .arg("--model")
        .arg(model)
        .args([
            "--backend",
            "ort-cuda,ort-tensorrt,ort-directml",
            "--warmup",
            "2",
            "--frames",
            "5",
            "--json",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let results = report["results"].as_array().unwrap();
    assert_eq!(
        results
            .iter()
            .map(|row| row["backend"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["ort-cuda", "ort-tensorrt", "ort-directml"]
    );
    for result in results {
        assert_eq!(result["status"], "ok");
        assert_eq!(result["audio_seconds"], 0.05);
        let rtf = result["rtf"].as_f64().unwrap();
        let fps = result["frames_per_second"].as_f64().unwrap();
        assert!((rtf * fps - 100.0).abs() < 1e-6);
    }
}
