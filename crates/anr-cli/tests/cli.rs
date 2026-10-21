use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Output, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

const DEADLINE: Duration = Duration::from_secs(60);

// Self-authored ONNX graph, not DFN3 weights or a noise-quality oracle:
// audio = (input + mean(states)) * (atten_lim_db / 60)
// next_states = states + mean(input). All ports are deliberately reordered.
fn model(dir: &Path) -> PathBuf {
    let path = dir.join("control.onnx");
    std::fs::write(&path, include_bytes!("fixtures/control.onnx")).unwrap();
    path
}

fn command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_anr"))
}

// Drain both output pipes concurrently with stdin writes. A stuck child is killed
// before panicking, including when an assertion unwinds through this owner.
struct Process {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Receiver<Vec<u8>>,
    stderr: Receiver<Vec<u8>>,
}

impl Process {
    fn spawn(command: &mut Command, stdout_prefix: Option<u64>) -> Self {
        Self::spawn_with_stdout(command, stdout_prefix, Stdio::piped())
    }

    fn spawn_with_stdout(
        command: &mut Command,
        stdout_prefix: Option<u64>,
        stdout_spec: Stdio,
    ) -> Self {
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(stdout_spec)
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take().unwrap();
        let (out_tx, out_rx) = mpsc::channel();
        let (err_tx, err_rx) = mpsc::channel();
        thread::spawn(move || {
            let mut bytes = Vec::new();
            if let Some(mut stdout) = stdout {
                let result = match stdout_prefix {
                    Some(limit) => stdout.take(limit).read_to_end(&mut bytes),
                    None => stdout.read_to_end(&mut bytes),
                };
                result.unwrap();
            }
            let _ = out_tx.send(bytes);
        });
        thread::spawn(move || {
            let mut bytes = Vec::new();
            let mut stderr = stderr;
            stderr.read_to_end(&mut bytes).unwrap();
            let _ = err_tx.send(bytes);
        });
        Self {
            child,
            stdin,
            stdout: out_rx,
            stderr: err_rx,
        }
    }

    fn finish(mut self) -> Output {
        drop(self.stdin.take());
        let start = Instant::now();
        let status = loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                break status;
            }
            assert!(start.elapsed() < DEADLINE, "CLI exceeded process deadline");
            thread::sleep(Duration::from_millis(10));
        };
        Output {
            status,
            stdout: self.stdout.recv_timeout(DEADLINE).unwrap(),
            stderr: self.stderr.recv_timeout(DEADLINE).unwrap(),
        }
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let start = Instant::now();
            while start.elapsed() < Duration::from_secs(5) {
                if self.child.try_wait().ok().flatten().is_some() {
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
        }
    }
}

fn run(command: &mut Command, input: Vec<u8>, fragmented: bool) -> Output {
    run_with_stdout_limit(command, input, fragmented, None)
}

fn run_with_stdout_limit(
    command: &mut Command,
    input: Vec<u8>,
    fragmented: bool,
    stdout_prefix: Option<u64>,
) -> Output {
    let mut process = Process::spawn(command, stdout_prefix);
    let mut stdin = process.stdin.take().unwrap();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let result = if fragmented {
            let mut offset = 0;
            let mut width = 1;
            let mut result = Ok(());
            while offset < input.len() {
                let end = (offset + width).min(input.len());
                if let Err(error) = stdin.write_all(&input[offset..end]) {
                    result = Err(error);
                    break;
                }
                offset = end;
                width = width % 3 + 1;
            }
            result
        } else {
            stdin.write_all(&input)
        };
        drop(stdin);
        let _ = tx.send(result);
    });
    let output = process.finish();
    // Early argument errors and BrokenPipe are allowed to close stdin early.
    let _ = rx
        .recv_timeout(DEADLINE)
        .expect("stdin writer exceeded deadline");
    output
}

fn assert_code(output: &Output, expected: i32) {
    assert_eq!(
        output.status.code(),
        Some(expected),
        "unexpected exit status; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(any(feature = "ort", feature = "gl"))]
fn assert_bytes(actual: &[u8], expected: &[u8]) {
    let mismatch = actual.iter().zip(expected).position(|(a, b)| a != b);
    assert!(
        actual.len() == expected.len() && mismatch.is_none(),
        "audio differs: actual {} bytes, expected {} bytes, first mismatch {:?}",
        actual.len(),
        expected.len(),
        mismatch
    );
}

fn bytes(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect()
}

fn raw_command(model: &Path, input: &Path, output: &Path, channels: usize) -> Command {
    let mut cmd = command();
    cmd.arg("denoise")
        .arg(input)
        .arg("--model")
        .arg(model)
        .args(["--input-format", "f32le", "--input-rate", "48000"])
        .arg("--input-channels")
        .arg(channels.to_string())
        .arg("--output")
        .arg(output)
        .args(["--output-format", "f32le", "--strength", "100", "--quiet"]);
    cmd
}

fn write_wav(path: &Path, rate: u32, channels: u16, samples: &[f32]) {
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .unwrap();
    for &sample in samples {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
}

#[cfg(any(feature = "ort", feature = "gl"))]
fn read_wav(path: &Path) -> (hound::WavSpec, Vec<f32>) {
    let mut reader = hound::WavReader::open(path).unwrap();
    let spec = reader.spec();
    let samples = if spec.sample_format == hound::SampleFormat::Float {
        reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    } else {
        let scale = (1_u32 << (spec.bits_per_sample - 1)) as f32;
        reader
            .samples::<i32>()
            .map(|sample| sample.unwrap() as f32 / scale)
            .collect()
    };
    (spec, samples)
}

fn directory_names(dir: &Path) -> Vec<std::ffi::OsString> {
    let mut names = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect::<Vec<_>>();
    names.sort();
    names
}

#[test]
fn inspect_json_reports_real_audio_without_a_model() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("音声 with spaces.wav");
    write_wav(&input, 16_000, 2, &vec![0.125; 320]);
    let output = run(
        command().arg("inspect").arg(&input).arg("--json"),
        vec![],
        false,
    );
    assert_code(&output, 0);
    let info: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(info["sample_rate"], 16_000);
    assert_eq!(info["channels"], 2);
    assert_eq!(info["frames"], 160);
    assert!((info["duration_seconds"].as_f64().unwrap() - 0.01).abs() < 1e-8);
    assert!(info["format"].as_str().is_some_and(|s| !s.is_empty()));
    assert!(info["codec"].as_str().is_some_and(|s| !s.is_empty()));
    assert!(info.get("track_id").is_some());
    assert!(info["track_id"].is_null() || info["track_id"].is_u64());
}

#[test]
fn backend_inventory_is_json_without_loading_a_model() {
    let output = run(command().args(["backends", "--json"]), vec![], false);
    assert_code(&output, 0);
    let inventory: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let backends = inventory["backends"].as_array().unwrap();
    let ort = backends
        .iter()
        .find(|backend| backend["name"] == "ort")
        .unwrap();
    assert_eq!(ort["compiled"], cfg!(feature = "ort"));
    let providers = ort["providers"].as_array().unwrap();
    let cpu = providers
        .iter()
        .find(|provider| provider["name"] == "cpu")
        .unwrap();
    assert_eq!(cpu["compiled"], cfg!(feature = "ort"));
    for provider in providers {
        assert!(provider.get("runtime_available").is_some());
        assert!(provider["runtime_available"].is_null());
    }
    let gl = backends
        .iter()
        .find(|backend| backend["name"] == "gl")
        .unwrap();
    assert_eq!(gl["compiled"], cfg!(feature = "gl"));
    assert!(gl["providers"].as_array().unwrap().is_empty());
    assert!(gl.get("runtime_available").is_some());
    assert!(gl["runtime_available"].is_null());
}

#[test]
fn invalid_raw_and_output_arguments_exit_two_without_creating_files() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    let output = dir.path().join("never-created.f32le");
    let cases: &[&[&str]] = &[
        &["--input-format", "f32le", "--input-channels", "1"],
        &["--input-format", "f32le", "--input-rate", "48000"],
        &[
            "--input-format",
            "f32le",
            "--input-rate",
            "0",
            "--input-channels",
            "1",
        ],
        &[
            "--input-format",
            "f32le",
            "--input-rate",
            "48000",
            "--input-channels",
            "0",
        ],
        &[
            "--input-format",
            "auto",
            "--input-rate",
            "48000",
            "--input-channels",
            "1",
        ],
        &["--strength", "NaN"],
        &["--strength", "inf"],
        &["--strength", "101"],
        &["--strength=-1"],
        &["--output-rate", "0"],
        &["--output-format", "f32le", "--wav-format", "pcm16"],
        &["--quiet", "--verbose"],
    ];
    for flags in cases {
        let result = run(
            command()
                .args(["denoise", "-"])
                .arg("--output")
                .arg(&output)
                .arg("--model")
                .arg(&model)
                .args(*flags),
            vec![],
            false,
        );
        assert_code(&result, 2);
        assert!(
            result.stdout.is_empty(),
            "argument errors contaminated stdout"
        );
        assert!(!output.exists(), "invalid arguments created an output");
    }
    for flags in [
        vec!["--output", "-"],
        vec!["--output", "-", "--output-format", "wav"],
    ] {
        let result = run(
            command()
                .args([
                    "denoise",
                    "-",
                    "--input-format",
                    "f32le",
                    "--input-rate",
                    "48000",
                    "--input-channels",
                    "1",
                ])
                .arg("--model")
                .arg(&model)
                .args(flags),
            vec![],
            false,
        );
        assert_code(&result, 2);
        assert!(result.stdout.is_empty());
    }
}

#[test]
fn unavailable_compiled_capability_fails_before_any_output() {
    let unavailable = if !cfg!(feature = "ort") {
        Some("cpu")
    } else {
        [
            ("cuda", cfg!(feature = "ort-cuda")),
            ("tensorrt", cfg!(feature = "ort-tensorrt")),
            (
                "directml",
                cfg!(all(feature = "ort-directml", target_os = "windows")),
            ),
            (
                "coreml",
                cfg!(all(feature = "ort-coreml", target_os = "macos")),
            ),
            ("xnnpack", cfg!(feature = "ort-xnnpack")),
        ]
        .into_iter()
        .find_map(|(provider, compiled)| (!compiled).then_some(provider))
    };
    // An all-provider build has no uncompiled provider to request.
    let Some(provider) = unavailable else { return };
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    let input = dir.path().join("source.f32le");
    let output = dir.path().join("output.f32le");
    std::fs::write(&input, bytes(&[0.125; 480])).unwrap();
    let before = directory_names(dir.path());
    let result = run(
        raw_command(&model, &input, &output, 1).args(["--provider", provider]),
        vec![],
        false,
    );
    assert_code(&result, 1);
    assert!(result.stdout.is_empty());
    assert!(!result.stderr.is_empty());
    assert_eq!(directory_names(dir.path()), before);
}

#[cfg(feature = "ort")]
#[test]
fn invalid_model_fails_before_touching_any_batch_output() {
    let dir = tempfile::tempdir().unwrap();
    let invalid_model = dir.path().join("invalid.onnx");
    let input = dir.path().join("input.wav");
    let output = dir.path().join("outputs");
    std::fs::create_dir(&output).unwrap();
    std::fs::write(&invalid_model, b"not an ONNX graph").unwrap();
    write_wav(&input, 48_000, 1, &[0.125; 480]);
    let result = run(
        command()
            .arg("denoise")
            .arg(&input)
            .arg("--output-dir")
            .arg(&output)
            .arg("--model")
            .arg(&invalid_model),
        vec![],
        false,
    );
    assert_code(&result, 1);
    assert!(result.stdout.is_empty());
    assert!(!result.stderr.is_empty());
    assert!(directory_names(&output).is_empty());
}

#[cfg(feature = "ort")]
#[test]
fn raw_pipe_preserves_state_across_frames_and_trims_the_final_padding() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    // The input crosses a typical 4096-frame decode block, and ends mid-frame.
    let input = bytes(&vec![0.03125; 480 * 10 + 17]);
    let expected = bytes(
        &(0..480 * 10 + 17)
            .map(|index| 0.03125 * (index / 480 + 1) as f32)
            .collect::<Vec<_>>(),
    );
    let result = run(
        &mut raw_command(&model, Path::new("-"), Path::new("-"), 1),
        input,
        true,
    );
    assert_code(&result, 0);
    assert_bytes(&result.stdout, &expected);
}

#[cfg(feature = "ort")]
#[test]
fn stereo_engines_do_not_leak_left_state_into_the_silent_right_channel() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    let mut input = Vec::new();
    let mut expected = Vec::new();
    for frame in 0..480 * 3 + 7 {
        input.extend([0.0625, 0.0]);
        expected.extend([0.0625 * (frame / 480 + 1) as f32, 0.0]);
    }
    let result = run(
        &mut raw_command(&model, Path::new("-"), Path::new("-"), 2),
        bytes(&input),
        false,
    );
    assert_code(&result, 0);
    assert_bytes(&result.stdout, &bytes(&expected));
}

#[cfg(feature = "ort")]
#[test]
fn strength_changes_the_model_output_instead_of_being_ignored() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    for (strength, expected) in [("0", 0.0), ("50", 0.0625), ("100", 0.125)] {
        let mut cmd = command();
        cmd.args([
            "denoise",
            "-",
            "--input-format",
            "f32le",
            "--input-rate",
            "48000",
            "--input-channels",
            "1",
            "--output",
            "-",
            "--output-format",
            "f32le",
            "--strength",
            strength,
        ])
        .arg("--model")
        .arg(&model);
        let result = run(&mut cmd, bytes(&[0.125; 480]), false);
        assert_code(&result, 0);
        assert_bytes(&result.stdout, &bytes(&[expected; 480]));
    }
}

#[cfg(feature = "ort")]
#[test]
fn wav_encodings_have_the_requested_headers_and_inferred_audio() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    for (format, bits, kind) in [
        ("pcm16", 16, hound::SampleFormat::Int),
        ("pcm24", 24, hound::SampleFormat::Int),
        ("float32", 32, hound::SampleFormat::Float),
    ] {
        let input = dir.path().join(format!("input-{format}.wav"));
        let output = dir.path().join(format!("output-{format}.wav"));
        // Each variant also proves that the decoder reads that WAV encoding.
        let mut writer = hound::WavWriter::create(
            &input,
            hound::WavSpec {
                channels: 2,
                sample_rate: 48_000,
                bits_per_sample: bits,
                sample_format: kind,
            },
        )
        .unwrap();
        for _ in 0..497 {
            if kind == hound::SampleFormat::Float {
                writer.write_sample(0.125_f32).unwrap();
                writer.write_sample(0.0_f32).unwrap();
            } else {
                writer.write_sample(1_i32 << (bits - 4)).unwrap();
                writer.write_sample(0_i32).unwrap();
            }
        }
        writer.finalize().unwrap();
        let result = run(
            command()
                .arg("denoise")
                .arg(&input)
                .arg("--output")
                .arg(&output)
                .arg("--model")
                .arg(&model)
                .args(["--strength", "100", "--wav-format", format]),
            vec![],
            false,
        );
        assert_code(&result, 0);
        assert!(result.stdout.is_empty());
        let (spec, samples) = read_wav(&output);
        assert_eq!(spec.channels, 2);
        assert_eq!(spec.sample_rate, 48_000);
        assert_eq!(spec.bits_per_sample, bits);
        assert_eq!(spec.sample_format, kind);
        assert_eq!(samples.len(), 994);
        let tolerance = if bits == 16 { 1.0 / 32768.0 } else { 1e-6 };
        for (frame, pair) in samples.as_chunks::<2>().0.iter().enumerate() {
            let expected = if frame < 480 { 0.125 } else { 0.25 };
            assert!(
                (pair[0] - expected).abs() <= tolerance,
                "{format} frame {frame}: got {}, expected {expected}",
                pair[0]
            );
            assert_eq!(pair[1], 0.0, "right channel is not silent at frame {frame}");
        }
    }
}

#[cfg(feature = "ort")]
#[test]
fn resampling_preserves_or_rounds_the_total_frame_count_without_losing_tail() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    for (index, input_rate, frames, output_rate, expected_frames) in [
        (0, 16_000, 1_003, None, 1_003),
        (1, 44_100, 4_097, None, 4_097),
        (2, 16_000, 1_003, Some(22_050), 1_382),
        (3, 44_100, 4_097, Some(48_000), 4_459),
        (4, 16_000, 7, Some(48_000), 21),
        (5, 44_100, 1, None, 1),
    ] {
        let input = dir.path().join(format!("input-{index}.wav"));
        let output = dir.path().join(format!("output-{index}.wav"));
        let samples = (0..frames)
            .flat_map(|_| [0.015625, 0.0])
            .collect::<Vec<_>>();
        write_wav(&input, input_rate, 2, &samples);
        let mut cmd = command();
        cmd.arg("denoise")
            .arg(&input)
            .arg("--output")
            .arg(&output)
            .arg("--model")
            .arg(&model)
            .args(["--strength", "100", "--wav-format", "float32"]);
        if let Some(rate) = output_rate {
            cmd.arg("--output-rate").arg(rate.to_string());
        }
        let result = run(&mut cmd, vec![], false);
        assert_code(&result, 0);
        let (spec, actual) = read_wav(&output);
        assert_eq!(spec.sample_rate, output_rate.unwrap_or(input_rate));
        assert_eq!(spec.channels, 2);
        assert_eq!(actual.len(), expected_frames * 2, "resampling case {index}");
        assert!(actual.iter().all(|sample| sample.is_finite()));
        assert!(actual.as_chunks::<2>().0.iter().all(|pair| pair[1] == 0.0));
        assert!(
            actual
                .as_chunks::<2>()
                .0
                .iter()
                .any(|pair| pair[0].abs() > 1e-6),
            "resampling case {index} discarded all audio"
        );
    }
}

#[cfg(feature = "ort")]
#[test]
fn batch_resets_between_files_and_keeps_successes_after_a_bad_input() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    let a = dir.path().join("first.wav");
    let bad = dir.path().join("broken.wav");
    let b = dir.path().join("last.wav");
    let output = dir.path().join("results");
    std::fs::create_dir(&output).unwrap();
    write_wav(&a, 48_000, 1, &vec![0.125; 977]);
    write_wav(&b, 48_000, 1, &vec![0.125; 977]);
    std::fs::write(&bad, b"not an audio container").unwrap();
    let result = run(
        command()
            .arg("denoise")
            .arg(&a)
            .arg(&bad)
            .arg(&b)
            .arg("--output-dir")
            .arg(&output)
            .arg("--model")
            .arg(&model)
            .args(["--output-format", "f32le", "--strength", "100"]),
        vec![],
        false,
    );
    assert_code(&result, 1);
    assert!(result.stdout.is_empty());
    assert!(!result.stderr.is_empty());
    let expected = bytes(
        &(0..977)
            .map(|i| 0.125 * (i / 480 + 1) as f32)
            .collect::<Vec<_>>(),
    );
    assert_bytes(
        &std::fs::read(output.join("first.f32le")).unwrap(),
        &expected,
    );
    assert_bytes(
        &std::fs::read(output.join("last.f32le")).unwrap(),
        &expected,
    );
    assert_eq!(
        directory_names(&output),
        ["first.f32le", "last.f32le"].map(std::ffi::OsString::from)
    );
}

#[cfg(feature = "ort")]
#[test]
fn overwrite_only_commits_after_success_and_never_truncates_the_old_file() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    let target = dir.path().join("existing.f32le");
    let original = b"existing target must survive unsuccessful processing";
    std::fs::write(&target, original).unwrap();
    let before = directory_names(dir.path());
    let rejected = run(
        &mut raw_command(&model, Path::new("-"), &target, 1),
        bytes(&[0.125; 480]),
        false,
    );
    assert_code(&rejected, 1);
    assert_bytes(&std::fs::read(&target).unwrap(), original);
    assert_eq!(directory_names(dir.path()), before);
    let mut truncated = bytes(&[0.125; 481]);
    truncated.push(1);
    let failed = run(
        raw_command(&model, Path::new("-"), &target, 1).arg("--overwrite"),
        truncated,
        true,
    );
    assert_code(&failed, 1);
    assert_bytes(&std::fs::read(&target).unwrap(), original);
    assert_eq!(directory_names(dir.path()), before);
    let replaced = run(
        raw_command(&model, Path::new("-"), &target, 1).arg("--overwrite"),
        bytes(&[0.125; 480]),
        false,
    );
    assert_code(&replaced, 0);
    assert_bytes(&std::fs::read(&target).unwrap(), &bytes(&[0.125; 480]));
    assert_eq!(directory_names(dir.path()), before);
}

#[cfg(feature = "ort")]
#[test]
fn truncated_stdin_is_an_error_and_removes_the_uncommitted_output() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    for (index, channels, tail) in [
        (0, 1, vec![0]),
        (1, 1, vec![0, 0, 0]),
        (2, 2, bytes(&[0.25])),
    ] {
        let target = dir.path().join(format!("truncated-{index}.f32le"));
        let before = directory_names(dir.path());
        let mut input = bytes(&vec![0.125; 480 * channels]);
        input.extend(tail);
        let result = run(
            &mut raw_command(&model, Path::new("-"), &target, channels),
            input,
            true,
        );
        assert_code(&result, 1);
        assert!(result.stdout.is_empty());
        assert!(!result.stderr.is_empty());
        assert_eq!(directory_names(dir.path()), before);
    }
}

#[cfg(feature = "ort")]
#[test]
fn a_consumer_closing_stdout_early_is_success_not_a_panic() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    let result = run_with_stdout_limit(
        &mut raw_command(&model, Path::new("-"), Path::new("-"), 1),
        bytes(&vec![0.0009765625; 480 * 1024]),
        false,
        Some(4),
    );
    assert_code(&result, 0);
    assert_bytes(&result.stdout, &0.0009765625_f32.to_le_bytes());
    assert!(
        result.stderr.is_empty(),
        "quiet BrokenPipe emitted a diagnostic"
    );
}

// Unix delivers SIGINT only to this child PID. Windows intentionally does not
// broadcast console control events into the test runner's shared console.
#[cfg(all(feature = "ort", unix))]
#[test]
fn interrupt_while_stdin_is_open_cleans_temp_and_preserves_the_target() {
    unsafe extern "C" {
        fn kill(pid: i32, signal: i32) -> i32;
    }
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    let target = dir.path().join("keep.f32le");
    let original = b"old completed output";
    std::fs::write(&target, original).unwrap();
    let before = directory_names(dir.path());
    let mut process = Process::spawn(
        raw_command(&model, Path::new("-"), &target, 1).arg("--overwrite"),
        None,
    );
    // Hold stdin open after several frames: cancellation must interrupt waiting
    // for further input, not depend on the producer eventually sending EOF.
    process
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&bytes(&[0.125; 960]))
        .unwrap();
    let start = Instant::now();
    while directory_names(dir.path()) == before {
        assert!(
            process.child.try_wait().unwrap().is_none(),
            "child exited before cancellation"
        );
        assert!(
            start.elapsed() < DEADLINE,
            "child did not create its transactional output"
        );
        thread::sleep(Duration::from_millis(10));
    }
    assert_bytes(&std::fs::read(&target).unwrap(), original);
    assert_eq!(unsafe { kill(process.child.id() as i32, 2) }, 0);
    // Keep the pipe writer alive until exit, so EOF cannot masquerade as cancel.
    let stdin = process.stdin.take().unwrap();
    let result = process.finish();
    drop(stdin);
    assert_code(&result, 130);
    assert_bytes(&std::fs::read(&target).unwrap(), original);
    assert_eq!(directory_names(dir.path()), before);
}

#[cfg(feature = "ort")]
#[test]
fn a_model_named_dash_is_still_protected_from_output_overwrite() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("-");
    let original_model = include_bytes!("fixtures/control.onnx");
    std::fs::write(&model, original_model).unwrap();
    let input = dir.path().join("voice.wav");
    write_wav(&input, 48_000, 1, &[0.125; 480]);
    let result = run(
        command()
            .current_dir(dir.path())
            .arg("denoise")
            .arg(&input)
            .args(["--model", "-"])
            .arg("--output")
            .arg(&model)
            .args(["--output-format", "wav", "--overwrite"]),
        vec![],
        false,
    );
    assert_code(&result, 2);
    assert!(std::fs::read(&model).unwrap() == original_model);
}

#[cfg(feature = "ort")]
#[test]
fn creating_a_parent_directory_cannot_turn_output_into_an_input_alias() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    let input = dir.path().join("voice.wav");
    write_wav(&input, 48_000, 1, &[0.125; 480]);
    let original_audio = std::fs::read(&input).unwrap();
    let output_directory = dir.path().join("new").join("..");
    let result = run(
        command()
            .arg("denoise")
            .arg(&input)
            .arg("--output-dir")
            .arg(&output_directory)
            .arg("--model")
            .arg(&model)
            .arg("--overwrite"),
        vec![],
        false,
    );
    assert_code(&result, 2);
    assert!(std::fs::read(&input).unwrap() == original_audio);
}

#[cfg(feature = "ort")]
#[test]
fn inference_failure_after_a_written_frame_preserves_the_existing_target() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("late-failure.onnx");
    std::fs::write(&model, include_bytes!("fixtures/late-failure.onnx")).unwrap();
    // First frame selects input[0]; the state then advances to 480 and makes
    // the next Gather out of bounds. Loading and the first inference are valid.
    let first = dir.path().join("first-frame.f32le");
    let result = run(
        &mut raw_command(&model, Path::new("-"), &first, 1),
        bytes(&[0.125; 480]),
        false,
    );
    assert_code(&result, 0);
    assert_bytes(&std::fs::read(&first).unwrap(), &bytes(&[0.25; 480]));
    let target = dir.path().join("existing.f32le");
    let original = b"previous completed audio";
    std::fs::write(&target, original).unwrap();
    let before = directory_names(dir.path());
    let result = run(
        raw_command(&model, Path::new("-"), &target, 1).arg("--overwrite"),
        bytes(&[0.125; 4096]),
        false,
    );
    assert_code(&result, 1);
    assert_bytes(&std::fs::read(&target).unwrap(), original);
    assert_eq!(directory_names(dir.path()), before);
}

#[cfg(feature = "ort")]
#[test]
fn publish_truncated_wav_preserves_the_existing_output() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    // The first case has no payload. The second decodes complete packets
    // before discovering that the final declared sample is missing.
    for (frames, missing) in [(480, 480), (4096, 1)] {
        let input = dir.path().join(format!("truncated-{frames}.wav"));
        write_wav(&input, 48_000, 1, &vec![0.125; frames]);
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&input)
            .unwrap();
        file.set_len(file.metadata().unwrap().len() - missing * 4)
            .unwrap();
        drop(file);
        let output = dir.path().join(format!("keep-{frames}.wav"));
        write_wav(&output, 48_000, 1, &[0.5; 480]);
        let original = std::fs::read(&output).unwrap();
        let before = directory_names(dir.path());
        let result = run(
            command()
                .arg("denoise")
                .arg(&input)
                .arg("--output")
                .arg(&output)
                .arg("--model")
                .arg(&model)
                .arg("--overwrite"),
            vec![],
            false,
        );
        assert_code(&result, 1);
        assert_bytes(&std::fs::read(&output).unwrap(), &original);
        assert_eq!(directory_names(dir.path()), before);
    }
}

#[cfg(feature = "ort")]
#[test]
fn publish_case_aliases_cannot_replace_inputs_or_models() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    let input = dir.path().join("VOICE.wav");
    write_wav(&input, 48_000, 1, &[0.125; 480]);
    let input_alias = dir.path().join("voice.wav");
    if !input_alias.exists() {
        // Exercised on Windows/macOS and on Linux with a case-insensitive TMPDIR.
        return;
    }
    let original_audio = std::fs::read(&input).unwrap();
    let original_model = std::fs::read(&model).unwrap();
    let before = directory_names(dir.path());
    for output in [input_alias, dir.path().join("CONTROL.ONNX")] {
        let result = run(
            command()
                .arg("denoise")
                .arg(&input)
                .arg("--model")
                .arg(&model)
                .arg("--output")
                .arg(output)
                .args(["--output-format", "wav", "--overwrite", "--strength", "0"]),
            vec![],
            false,
        );
        assert_code(&result, 2);
        assert_bytes(&std::fs::read(&input).unwrap(), &original_audio);
        assert_bytes(&std::fs::read(&model).unwrap(), &original_model);
        assert_eq!(directory_names(dir.path()), before);
    }
}

#[cfg(feature = "ort")]
#[test]
fn publish_batch_aliases_cannot_replace_an_earlier_result() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    let first_dir = dir.path().join("first");
    let second_dir = dir.path().join("second");
    let output_dir = dir.path().join("out");
    for path in [&first_dir, &second_dir, &output_dir] {
        std::fs::create_dir(path).unwrap();
    }
    let probe = output_dir.join("Probe");
    std::fs::write(&probe, b"case probe").unwrap();
    if !output_dir.join("probe").exists() {
        return;
    }
    std::fs::remove_file(probe).unwrap();
    let first = first_dir.join("A.wav");
    let second = second_dir.join("a.wav");
    write_wav(&first, 48_000, 1, &[0.125; 480]);
    write_wav(&second, 48_000, 1, &[0.375; 480]);
    let result = run(
        command()
            .arg("denoise")
            .arg(&first)
            .arg(&second)
            .arg("--output-dir")
            .arg(&output_dir)
            .arg("--model")
            .arg(&model)
            .args(["--overwrite", "--strength", "100"]),
        vec![],
        false,
    );
    if output_dir.join("A.wav").exists() {
        // A filesystem alias discovered only after the first commit must not
        // replace that completed result, even when --overwrite is explicit.
        assert_code(&result, 1);
        let (_, samples) = read_wav(&output_dir.join("A.wav"));
        assert_eq!(samples, vec![0.125; 480]);
        assert_eq!(directory_names(&output_dir).len(), 1);
    } else {
        // A platform that can detect the collision up front writes neither file.
        assert_code(&result, 2);
        assert!(directory_names(&output_dir).is_empty());
    }
}

#[cfg(feature = "ort")]
#[test]
fn publish_case_sensitive_filesystems_keep_distinct_names() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    let input = dir.path().join("voice.wav");
    let output = dir.path().join("VOICE.wav");
    write_wav(&input, 48_000, 1, &[0.125; 480]);
    if output.exists() {
        return;
    }
    let original = std::fs::read(&input).unwrap();
    let result = run(
        command()
            .arg("denoise")
            .arg(&input)
            .arg("--output")
            .arg(&output)
            .arg("--model")
            .arg(&model)
            .args(["--strength", "100"]),
        vec![],
        false,
    );
    assert_code(&result, 0);
    assert_bytes(&std::fs::read(&input).unwrap(), &original);
    let (_, samples) = read_wav(&output);
    assert_eq!(samples, vec![0.125; 480]);
}

#[test]
fn publish_json_broken_pipe_is_success() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("voice.wav");
    write_wav(&input, 48_000, 1, &[0.125; 480]);
    for mut cmd in [
        {
            let mut cmd = command();
            cmd.args(["backends", "--json"]);
            cmd
        },
        {
            let mut cmd = command();
            cmd.arg("inspect").arg(&input).arg("--json");
            cmd
        },
    ] {
        let (reader, writer) = std::io::pipe().unwrap();
        drop(reader);
        let result = Process::spawn_with_stdout(&mut cmd, None, writer.into()).finish();
        assert_code(&result, 0);
        assert!(result.stderr.is_empty());
    }
}

#[cfg(feature = "ort-load-dynamic")]
#[test]
fn publish_missing_dynamic_runtime_returns_error_without_touching_output() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    let input = dir.path().join("voice.f32le");
    let output = dir.path().join("keep.f32le");
    std::fs::write(&input, bytes(&[0.125; 480])).unwrap();
    let original = b"previous completed output";
    std::fs::write(&output, original).unwrap();
    let before = directory_names(dir.path());
    let providers: &[&str] = if cfg!(feature = "ort-xnnpack") {
        &["cpu", "xnnpack"]
    } else {
        &["cpu"]
    };
    for provider in providers {
        let result = run(
            raw_command(&model, &input, &output, 1)
                .env("ORT_DYLIB_PATH", dir.path().join("missing-runtime"))
                .args(["--provider", provider, "--overwrite"]),
            vec![],
            false,
        );
        assert_code(&result, 1);
        assert_bytes(&std::fs::read(&output).unwrap(), original);
        assert_eq!(directory_names(dir.path()), before);
    }
}

#[test]
fn gl_rejects_an_explicit_ort_provider_before_creating_output() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    let output = dir.path().join("never-created.f32le");
    for provider in ["cpu", "cuda"] {
        let result = run(
            raw_command(&model, Path::new("-"), &output, 1).args([
                "--backend",
                "gl",
                "--provider",
                provider,
            ]),
            vec![],
            false,
        );
        assert_code(&result, 2);
        assert!(result.stdout.is_empty());
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(
            error.contains("--provider") && error.contains("ORT"),
            "{error}"
        );
        assert!(!output.exists());
    }
}

#[cfg(not(feature = "gl"))]
#[test]
fn gl_without_feature_reports_backend_error_without_implicit_cpu_provider() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    let output = dir.path().join("never-created.f32le");
    let before = directory_names(dir.path());
    let result = run(
        raw_command(&model, Path::new("-"), &output, 1).args(["--backend", "gl"]),
        vec![],
        false,
    );
    assert_code(&result, 1);
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(error.contains("GL") && error.contains("未启用"), "{error}");
    assert!(!error.contains("provider cpu"), "{error}");
    assert_eq!(directory_names(dir.path()), before);
}

#[cfg(feature = "gl")]
#[test]
fn gl_rejects_a_nonmatching_onnx_model_before_touching_output() {
    let dir = tempfile::tempdir().unwrap();
    let model = model(dir.path());
    let output = dir.path().join("keep.f32le");
    std::fs::write(&output, b"existing output").unwrap();
    let before = directory_names(dir.path());
    let result = run(
        raw_command(&model, Path::new("-"), &output, 1)
            .args(["--backend", "gl", "--overwrite"])
            .env("ORT_DYLIB_PATH", dir.path().join("missing-ort-runtime")),
        vec![],
        false,
    );
    assert_code(&result, 1);
    assert!(result.stdout.is_empty());
    assert!(!result.stderr.is_empty());
    assert_eq!(std::fs::read(&output).unwrap(), b"existing output");
    assert_eq!(directory_names(dir.path()), before);
}

#[cfg(feature = "gl")]
fn gl_model() -> PathBuf {
    std::env::var_os("ANR_DFN3_MODEL")
        .map(PathBuf::from)
        .expect("set ANR_DFN3_MODEL to the supported original dfn3.onnx")
}

#[cfg(feature = "gl")]
fn decode_raw(data: &[u8]) -> Vec<f32> {
    assert_eq!(data.len() % 4, 0);
    data.as_chunks::<4>()
        .0
        .iter()
        .map(|sample| {
            let value = f32::from_le_bytes(*sample);
            assert!(value.is_finite(), "GL produced a non-finite audio sample");
            value
        })
        .collect()
}

#[cfg(feature = "gl")]
fn gl_signal() -> Vec<f32> {
    (0..1927)
        .map(|index| {
            let t = index as f32 / 48_000.0;
            0.15 * (std::f32::consts::TAU * 440.0 * t).sin()
                + 0.03 * (std::f32::consts::TAU * 3179.0 * t).sin()
        })
        .collect()
}

#[cfg(feature = "gl")]
#[test]
#[ignore = "requires ANR_DFN3_MODEL and a desktop OpenGL 4.3 compute driver"]
fn gl_raw_stereo_and_wav_preserve_length_and_independent_channel_state() {
    let dir = tempfile::tempdir().unwrap();
    let model = gl_model();
    let signal = gl_signal();
    let mono = run(
        raw_command(&model, Path::new("-"), Path::new("-"), 1)
            .args(["--backend", "gl"])
            .env("ORT_DYLIB_PATH", dir.path().join("missing-ort-runtime")),
        bytes(&signal),
        true,
    );
    assert_code(&mono, 0);
    let mono_samples = decode_raw(&mono.stdout);
    assert_eq!(mono_samples.len(), signal.len());
    assert!(mono_samples.iter().any(|sample| sample.abs() > 1e-7));
    let stereo: Vec<f32> = signal.iter().flat_map(|sample| [*sample, 0.0]).collect();
    let stereo_result = run(
        raw_command(&model, Path::new("-"), Path::new("-"), 2).args(["--backend", "gl"]),
        bytes(&stereo),
        true,
    );
    assert_code(&stereo_result, 0);
    let stereo_samples = decode_raw(&stereo_result.stdout);
    assert_eq!(stereo_samples.len(), signal.len() * 2);
    for (index, frame) in stereo_samples.as_chunks::<2>().0.iter().enumerate() {
        assert!((frame[0] - mono_samples[index]).abs() < 1e-6);
        assert!(
            frame[1].abs() < 1e-7,
            "left state leaked into silent right channel"
        );
    }
    let input = dir.path().join("stereo.wav");
    let output = dir.path().join("clean.wav");
    write_wav(&input, 48_000, 2, &stereo);
    let wav = run(
        command()
            .arg("denoise")
            .arg(&input)
            .arg("--model")
            .arg(&model)
            .arg("--output")
            .arg(&output)
            .args([
                "--backend",
                "gl",
                "--strength",
                "100",
                "--wav-format",
                "float32",
                "--verbose",
            ]),
        vec![],
        false,
    );
    assert_code(&wav, 0);
    let logs = String::from_utf8_lossy(&wav.stderr);
    assert!(logs.contains("backend=gl"), "{logs}");
    assert!(!logs.contains("provider="), "{logs}");
    let (spec, wav_samples) = read_wav(&output);
    assert_eq!(spec.sample_rate, 48_000);
    assert_eq!(spec.channels, 2);
    assert_eq!(wav_samples.len(), stereo_samples.len());
    for (actual, expected) in wav_samples.iter().zip(&stereo_samples) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[cfg(feature = "gl")]
#[test]
#[ignore = "requires ANR_DFN3_MODEL and a desktop OpenGL 4.3 compute driver"]
fn gl_batch_resets_state_and_strength_changes_real_audio() {
    let dir = tempfile::tempdir().unwrap();
    let model = gl_model();
    let signal = gl_signal();
    let first = dir.path().join("first.f32le");
    let second = dir.path().join("second.f32le");
    let output_dir = dir.path().join("batch");
    std::fs::write(&first, bytes(&signal)).unwrap();
    std::fs::write(&second, bytes(&signal)).unwrap();
    let batch = run(
        command()
            .arg("denoise")
            .arg(&first)
            .arg(&second)
            .arg("--model")
            .arg(&model)
            .arg("--output-dir")
            .arg(&output_dir)
            .args([
                "--backend",
                "gl",
                "--input-format",
                "f32le",
                "--input-rate",
                "48000",
                "--input-channels",
                "1",
                "--output-format",
                "f32le",
                "--strength",
                "100",
            ]),
        vec![],
        false,
    );
    assert_code(&batch, 0);
    let first_output = std::fs::read(output_dir.join("first.f32le")).unwrap();
    let second_output = std::fs::read(output_dir.join("second.f32le")).unwrap();
    assert_bytes(&first_output, &second_output);
    let processed = decode_raw(&first_output);
    assert_eq!(processed.len(), signal.len());
    let low_strength = run(
        command()
            .arg("denoise")
            .arg(&first)
            .arg("--model")
            .arg(&model)
            .args([
                "--backend",
                "gl",
                "--input-format",
                "f32le",
                "--input-rate",
                "48000",
                "--input-channels",
                "1",
                "--output",
                "-",
                "--output-format",
                "f32le",
                "--strength",
                "0",
            ]),
        vec![],
        false,
    );
    assert_code(&low_strength, 0);
    let low_samples = decode_raw(&low_strength.stdout);
    assert_eq!(low_samples.len(), signal.len());
    assert!(
        processed
            .iter()
            .zip(&low_samples)
            .any(|(high, low)| (high - low).abs() > 1e-7),
        "strength was ignored by the GL pipeline"
    );
}
