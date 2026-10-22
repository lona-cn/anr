use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn sdk(variable: &str, candidates: &[&str], prefix: &str) -> PathBuf {
    if let Some(value) = env::var_os(variable) {
        return PathBuf::from(value);
    }
    for candidate in candidates {
        let root = Path::new(candidate);
        if root.join("include").is_dir() {
            return root.to_owned();
        }
        if let Ok(entries) = fs::read_dir(root) {
            let mut roots: Vec<_> = entries
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .is_some_and(|n| n.to_string_lossy().starts_with(prefix))
                        && p.join("include").is_dir()
                })
                .collect();
            roots.sort();
            if let Some(path) = roots.pop() {
                return path;
            }
        }
    }
    panic!(
        "anr-tensorrt requires the native SDK: set {variable} to its root (containing include/); no SDK is downloaded or installed by this build"
    );
}

fn main() {
    for key in [
        "TENSORRT_ROOT",
        "CUDA_ROOT",
        "CUDA_PATH",
        "TENSORRT_LIB_DIR",
        "CUDA_LIB_DIR",
        "CXX",
        "CXXFLAGS",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    println!("cargo:rerun-if-changed=native/shim.cpp");
    println!("cargo:rerun-if-changed=native/shim.h");
    let windows = env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows";
    let trt = sdk(
        "TENSORRT_ROOT",
        &[
            "F:/Program Files",
            "C:/Program Files",
            "/opt/tensorrt",
            "/usr/local/TensorRT",
            "/usr",
        ],
        "TensorRT-",
    );
    let cuda = if let Some(root) = env::var_os("CUDA_ROOT").or_else(|| env::var_os("CUDA_PATH")) {
        PathBuf::from(root)
    } else {
        sdk(
            "CUDA_ROOT",
            &[
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA",
                "/usr/local/cuda",
            ],
            "v",
        )
    };
    for header in [
        trt.join("include/NvInfer.h"),
        cuda.join("include/cuda_runtime_api.h"),
        cuda.join("include/nvrtc.h"),
    ] {
        assert!(
            header.is_file(),
            "missing SDK header {}; check TENSORRT_ROOT/CUDA_ROOT",
            header.display()
        );
    }
    let trt_lib = env::var_os("TENSORRT_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if trt.join("lib").is_dir() {
                trt.join("lib")
            } else {
                trt.join("lib64")
            }
        });
    let cuda_lib = env::var_os("CUDA_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| cuda.join(if windows { "lib/x64" } else { "lib64" }));
    let out = fs::canonicalize(env::var_os("OUT_DIR").unwrap()).expect("resolve OUT_DIR");
    let source = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("native/shim.cpp");
    let filename = if windows {
        "anr_tensorrt_native.dll"
    } else {
        "libanr_tensorrt_native.so"
    };
    let shim = out.join(filename);
    let compiler = cc::Build::new().cpp(true).get_compiler();
    let mut command = compiler.to_command();
    command.current_dir(&out);
    if compiler.is_like_msvc() {
        let nvinfer = trt_lib.join(if trt_lib.join("nvinfer_10.lib").is_file() {
            "nvinfer_10.lib"
        } else {
            "nvinfer.lib"
        });
        command
            .args(["/nologo", "/LD", "/EHsc", "/std:c++17", "/MD", "/O2"])
            .arg(format!("/I{}", trt.join("include").display()))
            .arg(format!("/I{}", cuda.join("include").display()))
            .arg(&source)
            .arg("/link")
            .arg(nvinfer)
            .arg(cuda_lib.join("cudart.lib"))
            .arg(cuda_lib.join("nvrtc.lib"))
            .arg(cuda_lib.join("cuda.lib"))
            .arg(format!("/OUT:{}", shim.display()))
            .arg(format!(
                "/IMPLIB:{}",
                out.join("anr_tensorrt_native.lib").display()
            ));
    } else {
        command
            .args([
                "-std=c++17",
                "-O2",
                "-fPIC",
                "-shared",
                "-fvisibility=hidden",
            ])
            .arg("-I")
            .arg(trt.join("include"))
            .arg("-I")
            .arg(cuda.join("include"))
            .arg(&source)
            .arg("-L")
            .arg(&trt_lib)
            .arg("-L")
            .arg(&cuda_lib)
            .args(["-lnvinfer", "-lcudart", "-lnvrtc", "-lcuda"])
            .arg(format!("-Wl,-rpath,{}", trt_lib.display()))
            .arg(format!("-Wl,-rpath,{}", cuda_lib.display()))
            .arg("-o")
            .arg(&shim);
    }
    let result = command
        .output()
        .expect("start native C++ compiler; install MSVC Build Tools or a C++17 compiler");
    assert!(
        result.status.success(),
        "TensorRT/CUDA C++ shim compilation failed:\n{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    println!("cargo:rustc-env=ANR_TENSORRT_BUILT_SHIM={}", shim.display());
    println!("cargo:rustc-env=ANR_TENSORRT_SHIM_FILENAME={filename}");
    println!(
        "cargo:rustc-env=ANR_TENSORRT_BUILT_RUNTIME={}",
        trt.join("bin").display()
    );
    let cuda_bin = if cuda.join("bin/x64").is_dir() {
        cuda.join("bin/x64")
    } else {
        cuda.join("bin")
    };
    println!(
        "cargo:rustc-env=ANR_TENSORRT_BUILT_CUDA={}",
        cuda_bin.display()
    );
}
