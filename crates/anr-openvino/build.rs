use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn roots(path: &Path) -> Vec<PathBuf> {
    path.ancestors().take(5).map(Path::to_path_buf).collect()
}
fn library(dir: &Path, windows: bool, mac: bool) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    let mut paths: Vec<_> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
                return false;
            };
            if windows {
                name == "openvino.lib" || (name.starts_with("openvino-") && name.ends_with(".lib"))
            } else if mac {
                name == "libopenvino.dylib"
                    || (name.starts_with("libopenvino.") && name.ends_with(".dylib"))
            } else {
                name == "libopenvino.so" || name.starts_with("libopenvino.so.")
            }
        })
        .collect();
    paths.sort_by_key(|p| (p.file_name().unwrap().len(), p.clone()));
    paths.into_iter().next()
}
fn runtime_name(lib: &Path, windows: bool) -> String {
    let name = lib
        .file_name()
        .unwrap()
        .to_str()
        .expect("OpenVINO library filename must be UTF-8");
    if windows {
        format!("{}.dll", name.strip_suffix(".lib").unwrap())
    } else {
        name.to_owned()
    }
}
fn main() {
    for key in ["OPENVINO_ROOT", "OpenVINO_DIR", "CXX", "CXXFLAGS"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    println!("cargo:rerun-if-changed=native/openvino.cpp");
    let windows = env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows";
    let mac = env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos";
    let configured: Vec<PathBuf> = ["OPENVINO_ROOT", "OpenVINO_DIR"]
        .iter()
        .filter_map(|k| env::var_os(k).map(PathBuf::from))
        .collect();
    assert!(
        !configured.is_empty(),
        "anr-openvino requires the official OpenVINO C++ Runtime SDK. Set OPENVINO_ROOT to the SDK or official wheel's openvino directory (include/ and libs/), or OpenVINO_DIR to its CMake directory. No system installation is required."
    );
    let mut includes = None;
    let mut lib = None;
    for root in configured.iter().flat_map(|r| roots(r)) {
        for suffix in ["include", "runtime/include"] {
            let candidate = root.join(suffix);
            if candidate.join("openvino/openvino.hpp").is_file() {
                includes.get_or_insert(candidate);
            }
        }
        for suffix in [
            "libs",
            "lib",
            "lib/intel64",
            "runtime/lib/intel64/Release",
            "runtime/lib/intel64",
            "runtime/lib",
            "lib/intel64/Release",
        ] {
            if let Some(candidate) = library(&root.join(suffix), windows, mac) {
                lib.get_or_insert(candidate);
            }
        }
        if let Some(candidate) = library(&root, windows, mac) {
            lib.get_or_insert(candidate);
        }
    }
    let include = includes.expect("OpenVINO SDK headers not found: OPENVINO_ROOT must contain include/openvino/openvino.hpp or runtime/include/openvino/openvino.hpp; OpenVINO_DIR may point inside the SDK's cmake directory");
    let lib = lib.expect("OpenVINO Runtime link library not found under configured SDK: expected libs/openvino*.lib, runtime/lib/intel64/Release/openvino*.lib, or libopenvino.so/dylib (official wheel libs/ layout is supported)");
    // MSVC's /I option does not accept canonicalize's Windows \\?\ prefix.
    let include = std::path::absolute(include).expect("resolve OpenVINO include directory");
    let lib = std::path::absolute(lib).expect("resolve OpenVINO link library");
    let out = std::path::absolute(env::var_os("OUT_DIR").unwrap()).expect("resolve OUT_DIR");
    let source =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("native/openvino.cpp");
    let filename = if windows {
        "anr_openvino_native.dll"
    } else if mac {
        "libanr_openvino_native.dylib"
    } else {
        "libanr_openvino_native.so"
    };
    let shim = out.join(filename);
    let compiler = cc::Build::new().cpp(true).get_compiler();
    let mut command = compiler.to_command();
    command.current_dir(&out);
    if compiler.is_like_msvc() {
        command
            .args(["/nologo", "/LD", "/EHsc", "/std:c++17", "/MD", "/O2"])
            .arg(format!("/I{}", include.display()))
            .arg(&source)
            .arg("/link")
            .arg(&lib)
            .arg(format!("/OUT:{}", shim.display()))
            .arg(format!(
                "/IMPLIB:{}",
                out.join("anr_openvino_native.lib").display()
            ));
    } else {
        command
            .args(["-std=c++17", "-O2", "-fPIC", "-fvisibility=hidden"])
            .arg(if mac { "-dynamiclib" } else { "-shared" })
            .arg("-I")
            .arg(&include)
            .arg(&source)
            .arg(&lib)
            .arg(format!("-Wl,-rpath,{}", lib.parent().unwrap().display()))
            .arg("-o")
            .arg(&shim);
    }
    let result = command.output().expect("could not start C++ compiler; install a platform C++ toolchain (MSVC Build Tools on Windows)");
    assert!(
        result.status.success(),
        "OpenVINO C++ shim compilation failed using SDK {}:\n{}\n{}",
        include.display(),
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let runtime_filename = runtime_name(&lib, windows);
    let mut runtime = lib.parent().unwrap().join(&runtime_filename);
    if windows && !runtime.is_file() {
        for root in configured.iter().flat_map(|r| roots(r)) {
            for suffix in [
                "libs",
                "bin",
                "runtime/bin/intel64/Release",
                "runtime/bin/intel64",
                "runtime/bin",
            ] {
                let candidate = root.join(suffix).join(&runtime_filename);
                if candidate.is_file() {
                    runtime = candidate;
                    break;
                }
            }
        }
    }
    println!("cargo:rustc-env=ANR_OPENVINO_BUILT_SHIM={}", shim.display());
    println!("cargo:rustc-env=ANR_OPENVINO_SHIM_FILENAME={filename}");
    println!(
        "cargo:rustc-env=ANR_OPENVINO_BUILT_RUNTIME={}",
        runtime.display()
    );
    println!("cargo:rustc-env=ANR_OPENVINO_RUNTIME_FILENAME={runtime_filename}");
}
