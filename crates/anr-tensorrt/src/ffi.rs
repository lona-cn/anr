use anr_core::Result;
use anr_dfn3::{Executor, FRAME_SIZE, Model, backend};
use libloading::Library;
#[cfg(windows)]
use std::path::Path;
use std::{
    ffi::{CStr, CString, c_char, c_int, c_void},
    path::PathBuf,
    ptr::NonNull,
};

#[repr(C)]
struct Dispatch {
    shader: u32,
    groups: u32,
    params: [u32; 32],
}

type Create = unsafe extern "C" fn(
    *const f32,
    usize,
    u32,
    u32,
    u32,
    *const Dispatch,
    usize,
    usize,
    *const c_char,
    *mut c_char,
    usize,
) -> *mut c_void;
type Frame = unsafe extern "C" fn(
    *mut c_void,
    *const f32,
    *mut f32,
    f32,
    c_int,
    *mut c_char,
    usize,
) -> c_int;
type Destroy = unsafe extern "C" fn(*mut c_void);
type LastError = unsafe extern "C" fn() -> *const c_char;

#[cfg(windows)]
struct SearchDirectory(*mut c_void);
#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn AddDllDirectory(path: *const u16) -> *mut c_void;
    fn RemoveDllDirectory(cookie: *mut c_void) -> i32;
}
#[cfg(windows)]
impl SearchDirectory {
    fn add(path: &Path) -> Result<Self> {
        use std::os::windows::ffi::OsStrExt;
        let path = path.canonicalize().map_err(|e| {
            backend(format!(
                "TensorRT runtime directory {}: {e}",
                path.display()
            ))
        })?;
        let wide: Vec<_> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        let cookie = unsafe { AddDllDirectory(wide.as_ptr()) };
        if cookie.is_null() {
            return Err(backend(format!(
                "AddDllDirectory({}): {}",
                path.display(),
                std::io::Error::last_os_error()
            )));
        }
        Ok(Self(cookie))
    }
}
#[cfg(windows)]
impl Drop for SearchDirectory {
    fn drop(&mut self) {
        unsafe { RemoveDllDirectory(self.0) };
    }
}

/// Windows' restricted loader deliberately omits PATH. Admit only existing,
/// absolute PATH directories; empty/relative entries must not enable CWD DLL
/// search. Canonicalization also deduplicates aliases before registration.
#[cfg(windows)]
fn path_directories(value: &std::ffi::OsStr) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for path in std::env::split_paths(value) {
        if !path.is_absolute() || !path.is_dir() {
            continue;
        }
        if let Ok(path) = path.canonicalize()
            && !paths.contains(&path)
        {
            paths.push(path);
        }
    }
    paths
}

#[cfg(windows)]
fn runtime_directories(
    shim: &Path,
    executable: Option<&Path>,
    value: Option<&std::ffi::OsStr>,
) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    // Preserve DLL_LOAD_DIR/APPLICATION_DIR deployment without admitting the
    // implicit current working directory or relative PATH entries.
    for directory in shim
        .parent()
        .into_iter()
        .chain(executable.and_then(Path::parent))
    {
        if !directory.is_absolute() {
            continue;
        }
        if let Ok(path) = directory.canonicalize()
            && path.is_dir()
            && !paths.contains(&path)
        {
            paths.push(path);
        }
    }
    if let Some(value) = value {
        for directory in path_directories(value) {
            if !paths.contains(&directory) {
                paths.push(directory);
            }
        }
    }
    paths
}

#[cfg(windows)]
fn dependency_path(
    name: &str,
    explicit: Option<&Path>,
    path: &[PathBuf],
    built: &Path,
) -> Result<PathBuf> {
    if let Some(location) = explicit {
        let candidate = if location.is_file() {
            if !location
                .file_name()
                .is_some_and(|n| n.to_string_lossy().eq_ignore_ascii_case(name))
            {
                return Err(backend(format!(
                    "TensorRT explicit runtime {} does not name required DLL {name}",
                    location.display()
                )));
            }
            location.to_owned()
        } else {
            location.join(name)
        };
        return candidate.canonicalize().map_err(|e| backend(format!(
            "TensorRT explicit runtime must provide {name} at {}: {e}; PATH/SDK fallback is disabled for this override",
            candidate.display()
        )));
    }
    // Resolve each exact DLL basename from the actual shim import table. Do not
    // register every SDK on PATH: AddDllDirectory search ordering is unspecified
    // and could otherwise select a different TensorRT/CUDA installation.
    for directory in path
        .iter()
        .map(PathBuf::as_path)
        .chain(std::iter::once(built))
    {
        let candidate = directory.join(name);
        if candidate.is_file() {
            return candidate
                .canonicalize()
                .map_err(|e| backend(format!("TensorRT runtime {}: {e}", candidate.display())));
        }
    }
    Err(backend(format!(
        "TensorRT/CUDA runtime DLL {name} required by the native shim was not found in absolute PATH directories or {}",
        built.display()
    )))
}

/// Read the native shim's PE import names, not guessed CUDA version patterns.
/// This lets a relocated shim use precisely its linked runtime DLL ABI.
#[cfg(windows)]
fn shim_dependencies(path: &Path) -> Result<Vec<String>> {
    fn imports(bytes: &[u8]) -> Option<Vec<String>> {
        let word = |offset: usize| {
            Some(u16::from_le_bytes(
                bytes.get(offset..offset.checked_add(2)?)?.try_into().ok()?,
            ))
        };
        let dword = |offset: usize| {
            Some(u32::from_le_bytes(
                bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?,
            ))
        };
        if bytes.get(..2)? != b"MZ" {
            return None;
        }
        let pe = dword(0x3c)? as usize;
        if bytes.get(pe..pe.checked_add(4)?)? != b"PE\0\0" {
            return None;
        }
        let optional = pe.checked_add(24)?;
        let directory = match word(optional)? {
            0x20b => optional.checked_add(112)?,
            0x10b => optional.checked_add(96)?,
            _ => return None,
        };
        let import_rva = dword(directory.checked_add(8)?)?;
        let import_size = dword(directory.checked_add(12)?)? as usize;
        let sections = optional.checked_add(word(pe.checked_add(20)?)? as usize)?;
        let count = word(pe.checked_add(6)?)? as usize;
        let file_offset = |rva: u32| -> Option<usize> {
            for index in 0..count {
                let section = sections.checked_add(index.checked_mul(40)?)?;
                let start = dword(section.checked_add(12)?)?;
                let size = dword(section.checked_add(16)?)?;
                if let Some(delta) = rva.checked_sub(start)
                    && delta < size
                {
                    let offset =
                        (dword(section.checked_add(20)?)? as usize).checked_add(delta as usize)?;
                    if offset < bytes.len() {
                        return Some(offset);
                    }
                }
            }
            None
        };
        let table = file_offset(import_rva)?;
        let mut names = Vec::new();
        for index in 0..import_size / 20 {
            let entry = table.checked_add(index.checked_mul(20)?)?;
            let name_rva = dword(entry.checked_add(12)?)?;
            if name_rva == 0 {
                return Some(names);
            }
            let offset = file_offset(name_rva)?;
            let tail = bytes.get(offset..)?;
            let end = tail.iter().take(260).position(|byte| *byte == 0)?;
            let name = std::str::from_utf8(tail.get(..end)?)
                .ok()?
                .to_ascii_lowercase();
            if name.is_empty() || name.contains(['/', '\\', ':']) {
                return None;
            }
            if name.starts_with("nvinfer")
                || name.starts_with("cudart")
                || name.starts_with("nvrtc")
            {
                if !name.ends_with(".dll") {
                    return None;
                }
                names.push(name);
            }
        }
        None
    }
    let bytes = std::fs::read(path).map_err(|e| {
        backend(format!(
            "read TensorRT shim imports {}: {e}",
            path.display()
        ))
    })?;
    let names = imports(&bytes).ok_or_else(|| {
        backend(format!(
            "invalid TensorRT native shim PE import table: {}",
            path.display()
        ))
    })?;
    if names.is_empty() {
        return Err(backend(
            "TensorRT native shim has no TensorRT/CUDA runtime imports",
        ));
    }
    Ok(names)
}

#[cfg(windows)]
fn runtime_dependencies(shim: &Path) -> Result<(Vec<Library>, Vec<SearchDirectory>)> {
    let names = shim_dependencies(shim)?;
    let executable = std::env::current_exe().ok();
    let path = runtime_directories(
        shim,
        executable.as_deref(),
        std::env::var_os("PATH").as_deref(),
    );
    let trt = std::env::var_os("ANR_TENSORRT_RUNTIME").map(PathBuf::from);
    let cuda = std::env::var_os("CUDA_ROOT")
        .or_else(|| std::env::var_os("CUDA_PATH"))
        .map(|root| {
            let root = PathBuf::from(root);
            if root.join("bin/x64").is_dir() {
                root.join("bin/x64")
            } else {
                root.join("bin")
            }
        });
    let mut files = Vec::new();
    let mut directories = Vec::new();
    for name in names {
        let (explicit, built) = if name.starts_with("nvinfer") {
            (
                trt.as_deref(),
                Path::new(env!("ANR_TENSORRT_BUILT_RUNTIME")),
            )
        } else {
            (cuda.as_deref(), Path::new(env!("ANR_TENSORRT_BUILT_CUDA")))
        };
        let file = dependency_path(&name, explicit, &path, built)?;
        let parent = file
            .parent()
            .ok_or_else(|| backend("TensorRT runtime DLL has no parent directory"))?
            .to_owned();
        if !directories.contains(&parent) {
            directories.push(parent);
        }
        files.push(file);
    }
    // Cookies cover both immediate dependencies and late builder-resource /
    // NVRTC-builtins loads from these same selected SDK directories.
    let cookies = directories
        .iter()
        .map(|path| SearchDirectory::add(path))
        .collect::<Result<Vec<_>>>()?;
    let mut libraries = Vec::new();
    for file in files {
        let library = unsafe {
            libloading::os::windows::Library::load_with_flags(&file, 0x0000_0100 | 0x0000_1000)
        }
        .map(Library::from)
        .map_err(|e| {
            backend(format!(
                "load exact TensorRT/CUDA runtime {}: {e}",
                file.display()
            ))
        })?;
        libraries.push(library);
    }
    Ok((libraries, cookies))
}

struct Api {
    create: Create,
    frame: Frame,
    destroy: Destroy,
    last_error: LastError,
    // The shim must outlive the native object and its function pointers. The
    // Windows directory cookies also remain valid for deferred builder DLLs.
    _library: Library,
    #[cfg(windows)]
    _dependencies: Vec<Library>,
    #[cfg(windows)]
    _directories: Vec<SearchDirectory>,
}

impl Api {
    fn load() -> Result<Self> {
        let explicit = std::env::var_os("ANR_TENSORRT_SHIM").map(PathBuf::from);
        let mut candidates = Vec::new();
        if let Some(path) = explicit {
            candidates.push(path);
        } else {
            if let Ok(executable) = std::env::current_exe()
                && let Some(parent) = executable.parent()
            {
                candidates.push(parent.join(env!("ANR_TENSORRT_SHIM_FILENAME")));
            }
            candidates.push(PathBuf::from(env!("ANR_TENSORRT_BUILT_SHIM")));
        }
        let mut failures = String::new();
        for candidate in candidates {
            let candidate = candidate.canonicalize().unwrap_or(candidate);
            #[cfg(windows)]
            let (dependencies, directories) = match runtime_dependencies(&candidate) {
                Ok(dependencies) => dependencies,
                Err(error) => {
                    failures.push_str(&format!("{}: {error}\n", candidate.display()));
                    continue;
                }
            };
            // Loading happens only on explicit backend construction, never at
            // process startup. Dependent SDK libraries may therefore be absent
            // without preventing the CLI from reporting other backend results.
            #[cfg(windows)]
            let loaded = unsafe {
                libloading::os::windows::Library::load_with_flags(
                    &candidate,
                    0x0000_0100 | 0x0000_1000,
                )
            }
            .map(Library::from);
            #[cfg(not(windows))]
            let loaded = unsafe { Library::new(&candidate) };
            let library = match loaded {
                Ok(library) => library,
                Err(error) => {
                    failures.push_str(&format!("{}: {error}\n", candidate.display()));
                    continue;
                }
            };
            unsafe {
                let create = *library
                    .get::<Create>(b"anr_trt_create\0")
                    .map_err(anr_core::Error::backend)?;
                let frame = *library
                    .get::<Frame>(b"anr_trt_frame\0")
                    .map_err(anr_core::Error::backend)?;
                let destroy = *library
                    .get::<Destroy>(b"anr_trt_destroy\0")
                    .map_err(anr_core::Error::backend)?;
                let last_error = *library
                    .get::<LastError>(b"anr_trt_last_error\0")
                    .map_err(anr_core::Error::backend)?;
                return Ok(Self {
                    create,
                    frame,
                    destroy,
                    last_error,
                    _library: library,
                    #[cfg(windows)]
                    _dependencies: dependencies,
                    #[cfg(windows)]
                    _directories: directories,
                });
            }
        }
        Err(backend(format!(
            "TensorRT/CUDA runtime unavailable. Set ANR_TENSORRT_SHIM to the private native shim; put TensorRT/CUDA DLLs on PATH or set ANR_TENSORRT_RUNTIME and CUDA_ROOT (Linux: loader search path).\n{failures}"
        )))
    }
}

pub(crate) struct Gpu {
    instance: NonNull<c_void>,
    error: Box<[c_char; 65536]>,
    api: Api,
}

impl Gpu {
    pub(crate) fn new(model: Model) -> Result<Self> {
        let api = Api::load()?;
        let dispatches: Vec<_> = model
            .dispatches
            .iter()
            .map(|d| Dispatch {
                shader: d.shader.index() as u32,
                groups: d.groups,
                params: d.params,
            })
            .collect();
        let source = CString::new(anr_dfn3::shaders::cuda()).map_err(anr_core::Error::backend)?;
        let mut error = Box::new([0; 65536]);
        // Creation builds both engines synchronously and retains no borrowed
        // host pointers. The owned packed model is released after this call.
        let pointer = unsafe {
            (api.create)(
                model.weights.as_ptr(),
                model.weights.len(),
                model.scratch_len,
                model.input,
                model.output,
                dispatches.as_ptr(),
                dispatches.len(),
                model.attenuation_dispatch,
                source.as_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let instance = NonNull::new(pointer).ok_or_else(|| diagnostic(&api, &error[..]))?;
        Ok(Self {
            instance,
            error,
            api,
        })
    }
}

fn diagnostic(api: &Api, error: &[c_char]) -> anr_core::Error {
    // Thread-local native storage retains the entire logger diagnostic. The
    // fixed emergency buffer remains available if allocating that storage failed.
    let pointer = unsafe { (api.last_error)() };
    let full = unsafe { CStr::from_ptr(pointer) };
    let text = if full.is_empty() {
        unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy()
    } else {
        full.to_string_lossy()
    };
    backend(format!("native TensorRT: {text}"))
}

impl Executor for Gpu {
    fn frame(
        &mut self,
        input: &[f32; FRAME_SIZE],
        output: &mut [f32; FRAME_SIZE],
        attenuation: f32,
        reset: bool,
    ) -> Result<()> {
        let status = unsafe {
            (self.api.frame)(
                self.instance.as_ptr(),
                input.as_ptr(),
                output.as_mut_ptr(),
                attenuation,
                i32::from(reset),
                self.error.as_mut_ptr(),
                self.error.len(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(diagnostic(&self.api, &self.error[..]))
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe { (self.api.destroy)(self.instance.as_ptr()) };
    }
}

#[cfg(all(test, windows))]
mod loader_tests {
    use super::*;

    #[test]
    fn path_directories_exclude_empty_relative_and_duplicate_entries() {
        let directory = tempfile::tempdir().unwrap();
        let value = std::env::join_paths([
            Path::new(""),
            Path::new("."),
            Path::new("relative/runtime"),
            directory.path(),
            directory.path(),
        ])
        .unwrap();
        assert_eq!(
            path_directories(&value),
            vec![directory.path().canonicalize().unwrap()]
        );
    }

    #[test]
    fn colocated_runtimes_precede_path_and_build_machine_directories() {
        let root = tempfile::tempdir().unwrap();
        let shim = root.path().join("shim");
        let application = root.path().join("application");
        let path = root.path().join("path");
        let absent = root.path().join("absent-sdk");
        let name = "nvinfer_10.dll";
        for directory in [&shim, &application, &path] {
            std::fs::create_dir(directory).unwrap();
            std::fs::write(directory.join(name), []).unwrap();
        }
        let value = std::env::join_paths([&path]).unwrap();
        let directories = runtime_directories(
            &shim.join("native.dll"),
            Some(&application.join("anr.exe")),
            Some(&value),
        );
        let selected = || dependency_path(name, None, &directories, &absent).unwrap();
        assert_eq!(selected(), shim.join(name).canonicalize().unwrap());
        std::fs::remove_file(shim.join(name)).unwrap();
        assert_eq!(selected(), application.join(name).canonicalize().unwrap());
        std::fs::remove_file(application.join(name)).unwrap();
        assert_eq!(selected(), path.join(name).canonicalize().unwrap());
    }

    #[test]
    fn explicit_runtime_override_does_not_fall_back_to_path() {
        let empty = tempfile::tempdir().unwrap();
        let built = Path::new(env!("ANR_TENSORRT_BUILT_RUNTIME"));
        let names = shim_dependencies(Path::new(env!("ANR_TENSORRT_BUILT_SHIM"))).unwrap();
        let name = names
            .iter()
            .find(|name| name.starts_with("nvinfer"))
            .unwrap();
        assert!(dependency_path(name, Some(empty.path()), &[built.to_owned()], built).is_err());
    }

    #[test]
    fn path_only_finds_the_exact_linked_runtime() {
        let built = Path::new(env!("ANR_TENSORRT_BUILT_RUNTIME"));
        let names = shim_dependencies(Path::new(env!("ANR_TENSORRT_BUILT_SHIM"))).unwrap();
        let name = names
            .iter()
            .find(|name| name.starts_with("nvinfer"))
            .unwrap();
        let absent = tempfile::tempdir().unwrap();
        let actual = dependency_path(name, None, &[built.to_owned()], absent.path()).unwrap();
        assert_eq!(actual, built.join(name).canonicalize().unwrap());
        assert!(
            dependency_path(
                "nvinfer_nonexistent_abi.dll",
                None,
                &[built.to_owned()],
                absent.path()
            )
            .is_err()
        );
    }
}
