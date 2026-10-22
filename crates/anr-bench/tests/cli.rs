#![cfg(any(not(feature = "ort"), feature = "ort-load-dynamic"))]

use std::{path::PathBuf, process::Command};

#[test]
fn missing_ort_library_returns_all_provider_errors_without_aborting_the_process() {
    let missing = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("missing-ort-runtime.dll");
    let output = Command::new(env!("CARGO_BIN_EXE_anr-bench"))
        .args([
            "--backend",
            "ort-cuda,ort-tensorrt,ort-directml",
            "--warmup",
            "0",
            "--frames",
            "1",
            "--json",
        ])
        .env("ORT_DYLIB_PATH", missing)
        .output()
        .unwrap();

    assert_eq!(
        output.status.code(),
        Some(1),
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
        assert_eq!(result["status"], "error");
        assert!(
            result.get("mean_ms").is_none(),
            "a failed provider must not report performance"
        );
    }
}

#[test]
fn native_backend_failures_keep_order_without_reporting_inference_timings() {
    let missing = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("missing-native-model.onnx");
    let output = Command::new(env!("CARGO_BIN_EXE_anr-bench"))
        .args([
            "--backend",
            "cuda,tensorrt,openvino,opencl",
            "--warmup",
            "0",
            "--frames",
            "1",
            "--json",
            "--model",
        ])
        .arg(missing)
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(1),
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
        ["cuda", "tensorrt", "openvino", "opencl"]
    );
    for result in results {
        assert_eq!(result["status"], "error");
        assert!(result.get("mean_ms").is_none());
        assert!(result.get("frames_per_second").is_none());
    }
}
