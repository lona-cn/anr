use anr_core::NoiseReduction;
use anr_ort::{DFN3, ort};
use std::{
    path::PathBuf,
    process::{Child, Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};

const CHILD_CASE: &str = "ANR_RUNTIME_TEST_CHILD";
#[cfg(feature = "load-dynamic")]
const VALID_LIBRARY: &str = "ANR_RUNTIME_TEST_VALID_LIBRARY";
const DEADLINE: Duration = Duration::from_secs(60);

#[cfg(feature = "load-dynamic")]
fn missing_library() -> PathBuf {
    std::env::temp_dir().join(format!(
        "anr-nonexistent-runtime-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

struct Process(Child);

impl Process {
    fn finish(&mut self) -> ExitStatus {
        let start = Instant::now();
        loop {
            if let Some(status) = self.0.try_wait().unwrap() {
                return status;
            }
            assert!(
                start.elapsed() < DEADLINE,
                "runtime child exceeded deadline"
            );
            thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        if self.0.try_wait().ok().flatten().is_none() {
            let _ = self.0.kill();
            let start = Instant::now();
            while start.elapsed() < Duration::from_secs(5) {
                if self.0.try_wait().ok().flatten().is_some() {
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
        }
    }
}

fn run_case(name: &str, case: impl FnOnce()) {
    if let Ok(child_case) = std::env::var(CHILD_CASE) {
        assert_eq!(child_case, name);
        case();
        return;
    }

    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", name, "--nocapture", "--test-threads=1"])
        .env(CHILD_CASE, name)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    #[cfg(feature = "load-dynamic")]
    {
        command.env_remove(VALID_LIBRARY);
        if let Some(path) = std::env::var_os("ORT_DYLIB_PATH") {
            command.env(VALID_LIBRARY, path);
        }
        if name != "loaded_runtime_and_host_environment_survive_invalid_path" {
            command.env("ORT_DYLIB_PATH", missing_library());
        }
    }
    let mut process = Process(command.spawn().unwrap());
    let status = process.finish();
    assert!(status.success(), "{name} child failed: {status}");
}

#[cfg(feature = "load-dynamic")]
#[test]
fn failed_dynamic_load_can_be_repeated() {
    run_case("failed_dynamic_load_can_be_repeated", || {
        assert_missing_library_error();
        assert_missing_library_error();
    });
}

#[cfg(feature = "load-dynamic")]
#[test]
fn failed_dynamic_load_can_recover_and_run_inference() {
    run_case("failed_dynamic_load_can_recover_and_run_inference", || {
        assert_missing_library_error();
        restore_library_path();
        anr_ort::initialize_runtime().unwrap();
        let environment = ort::init().build().unwrap();
        infer(&environment);
    });
}

#[cfg(feature = "load-dynamic")]
#[test]
fn loaded_runtime_and_host_environment_survive_invalid_path() {
    run_case(
        "loaded_runtime_and_host_environment_survive_invalid_path",
        || {
            anr_ort::initialize_runtime().unwrap();
            // No environment, session, or worker pools exist before this mutation.
            unsafe { std::env::set_var("ORT_DYLIB_PATH", missing_library()) };
            let environment = ort::init()
                .with_name("anr-host-environment")
                .build()
                .unwrap();
            anr_ort::initialize_runtime().unwrap();
            infer(&environment);
        },
    );
}

#[cfg(not(feature = "load-dynamic"))]
#[test]
fn cpu_inference_does_not_require_device_discovery() {
    run_case("cpu_inference_does_not_require_device_discovery", || {
        // Install one capability failure before ort initializes its global API.
        // Clone the table: never mutate the runtime's read-only API storage.
        let base = unsafe { ort::sys::OrtGetApiBase() };
        assert!(!base.is_null());
        let api = unsafe { ((*base).GetApi)(ort::sys::ORT_API_VERSION) };
        assert!(!api.is_null());
        let mut api = unsafe { &*api }.clone();
        api.GetEpDevices = discovery_unsupported;
        assert!(ort::set_api(api));
        anr_ort::initialize_runtime().unwrap();
        let environment = ort::init().build().unwrap();
        infer(&environment);
    });
}

fn infer(environment: &ort::environment::Environment) {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/stateful.onnx");
    let mut reducer = DFN3::from_file(environment, fixture).unwrap();
    reducer.set_factor(10.0);
    let mut output = [0.0; 480];
    reducer.run(48_000, &[1.0; 480], &mut output).unwrap();
    assert_eq!(output, [2.0; 480]);
    reducer.run(48_000, &[1.0; 480], &mut output).unwrap();
    assert_eq!(output, [8.0; 480]);
}

#[cfg(feature = "load-dynamic")]
fn assert_missing_library_error() {
    use std::error::Error;

    let error = anr_ort::initialize_runtime().unwrap_err();
    let source = error
        .source()
        .and_then(|source| source.downcast_ref::<ort::LoadDynamicError>())
        .expect("the original dynamic loader error must remain in the source chain");
    match source {
        ort::LoadDynamicError::Dlopen { path, .. } => {
            assert_eq!(
                path.as_os_str(),
                std::env::var_os("ORT_DYLIB_PATH").unwrap()
            );
        }
        other => panic!("expected a missing-library load failure, got {other:?}"),
    }
}

#[cfg(feature = "load-dynamic")]
fn restore_library_path() {
    // Only called in an isolated child selecting exactly one test, before ORT
    // starts any workers. The parent test process never changes its environment.
    unsafe {
        if let Some(path) = std::env::var_os(VALID_LIBRARY) {
            std::env::set_var("ORT_DYLIB_PATH", path);
        } else {
            std::env::remove_var("ORT_DYLIB_PATH");
        }
    }
}

#[cfg(not(feature = "load-dynamic"))]
unsafe extern "system" fn discovery_unsupported(
    _environment: *const ort::sys::OrtEnv,
    _devices: *mut *const *const ort::sys::OrtEpDevice,
    _count: *mut usize,
) -> ort::sys::OrtStatusPtr {
    // Session creation and inference still delegate to the genuine runtime.
    unsafe {
        (ort::api().CreateStatus)(
            ort::sys::OrtErrorCode::ORT_NOT_IMPLEMENTED,
            c"device discovery is unavailable".as_ptr(),
        )
    }
}
