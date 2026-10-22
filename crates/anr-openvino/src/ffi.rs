use crate::Device;
use anr_core::Result;
use anr_dfn3::{Executor, FRAME_SIZE, backend};
use libloading::Library;
use std::{
    ffi::{CStr, CString, c_char, c_void},
    path::PathBuf,
    ptr::NonNull,
};

type Create = unsafe extern "C" fn(
    *const u8,
    usize,
    *const c_char,
    *mut *mut c_void,
    *mut c_char,
    usize,
    *mut c_char,
    usize,
) -> i32;
type Frame =
    unsafe extern "C" fn(*mut c_void, *const f32, *mut f32, f32, i32, *mut c_char, usize) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void);

fn paths(variable: &str, filename: &str, built: &str) -> Vec<PathBuf> {
    if let Some(path) = std::env::var_os(variable) {
        return vec![PathBuf::from(path)];
    }
    let mut result = Vec::new();
    if let Ok(executable) = std::env::current_exe()
        && let Some(parent) = executable.parent()
    {
        result.push(parent.join(filename));
    }
    result.push(PathBuf::from(built));
    result
}

unsafe fn open(path: &PathBuf) -> std::result::Result<Library, libloading::Error> {
    #[cfg(windows)]
    {
        // Restrict dependent DLL discovery to this official SDK/deployment
        // directory and the normal safe system directories, not process CWD.
        const LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: u32 = 0x00000100;
        const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x00001000;
        unsafe {
            libloading::os::windows::Library::load_with_flags(
                path,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
            )
            .map(Into::into)
        }
    }
    #[cfg(not(windows))]
    unsafe {
        Library::new(path)
    }
}

fn load(variable: &str, filename: &str, built: &str) -> Result<Library> {
    let candidates = paths(variable, filename, built);
    let mut errors = Vec::new();
    for candidate in candidates {
        let path = if candidate.is_absolute() {
            candidate
        } else {
            std::env::current_dir()
                .map_err(anr_core::Error::backend)?
                .join(candidate)
        };
        // SAFETY: these paths are trusted application deployment/configuration,
        // never model data. Handles remain alive until native objects are gone.
        match unsafe { open(&path) } {
            Ok(library) => return Ok(library),
            Err(error) => errors.push(format!("{}: {error}", path.display())),
        }
    }
    Err(backend(format!(
        "cannot load native OpenVINO library; set {variable} to its shared-library path and install/deploy the official Runtime, ONNX frontend, and selected device plugin. No fallback is attempted. Tried: {}",
        errors.join("; ")
    )))
}

pub(crate) struct Native {
    handle: NonNull<c_void>,
    frame: Frame,
    destroy: Destroy,
    error: [c_char; 8192],
    execution_device: String,
    // Rust fields drop in declaration order, after Drop destroys the request.
    _shim: Library,
    _runtime: Library,
}
impl Native {
    pub(crate) fn new(bytes: Vec<u8>, device: Device) -> Result<Self> {
        let runtime = load(
            "ANR_OPENVINO_RUNTIME",
            env!("ANR_OPENVINO_RUNTIME_FILENAME"),
            env!("ANR_OPENVINO_BUILT_RUNTIME"),
        )?;
        let shim = load(
            "ANR_OPENVINO_SHIM",
            env!("ANR_OPENVINO_SHIM_FILENAME"),
            env!("ANR_OPENVINO_BUILT_SHIM"),
        )?;
        // SAFETY: the private shim exports exactly these versioned-by-build C
        // signatures. Function pointers are used only while its handle is live.
        let (create, frame, destroy) = unsafe {
            (
                *shim
                    .get::<Create>(b"anr_ov_create\0")
                    .map_err(anr_core::Error::backend)?,
                *shim
                    .get::<Frame>(b"anr_ov_frame\0")
                    .map_err(anr_core::Error::backend)?,
                *shim
                    .get::<Destroy>(b"anr_ov_destroy\0")
                    .map_err(anr_core::Error::backend)?,
            )
        };
        let selected = CString::new(device.to_string()).expect("device is ASCII without NUL");
        let mut error = [0; 8192];
        let mut actual = [0; 256];
        let mut handle = std::ptr::null_mut();
        // SAFETY: all pointers are live and have the exact lengths required by
        // the C ABI. Create copies/imports the model before returning.
        let status = unsafe {
            create(
                bytes.as_ptr(),
                bytes.len(),
                selected.as_ptr(),
                &mut handle,
                actual.as_mut_ptr(),
                actual.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(native_error(&error));
        }
        let handle = NonNull::new(handle)
            .ok_or_else(|| backend("OpenVINO initialization returned a null native handle"))?;
        // SAFETY: create guarantees a terminated execution-device string on success.
        let execution_device = unsafe { CStr::from_ptr(actual.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        Ok(Self {
            handle,
            frame,
            destroy,
            error,
            execution_device,
            _shim: shim,
            _runtime: runtime,
        })
    }
    pub(crate) fn execution_device(&self) -> &str {
        &self.execution_device
    }
}
fn native_error(buffer: &[c_char; 8192]) -> anr_core::Error {
    // SAFETY: the zero-initialized buffer is always NUL-terminated by the shim.
    backend(
        unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned(),
    )
}
impl Executor for Native {
    fn frame(
        &mut self,
        input: &[f32; FRAME_SIZE],
        output: &mut [f32; FRAME_SIZE],
        attenuation: f32,
        reset: bool,
    ) -> Result<()> {
        // SAFETY: handle is uniquely worker-owned, input/output are one full
        // frame, and the synchronous native call finishes before returning.
        let status = unsafe {
            (self.frame)(
                self.handle.as_ptr(),
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
            Err(native_error(&self.error))
        }
    }
}
impl Drop for Native {
    fn drop(&mut self) {
        // SAFETY: initialized exactly once and no asynchronous work is issued;
        // destructor runs on the owning worker before library handles unload.
        unsafe { (self.destroy)(self.handle.as_ptr()) };
    }
}
