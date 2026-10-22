use anr_core::Result;
use anr_dfn3::backend;
use cudarc::nvrtc::Ptx;
use libloading::Library;
use std::ffi::{CStr, CString, c_char, c_int, c_void};

// Keep this list in step with the cudarc calls in gpu.rs, including their
// cleanup and error-formatting paths. Resolve everything before cudarc's
// panic-on-missing-symbol loader is allowed to run (also for panic=abort).
const DRIVER_SYMBOLS: &[&[u8]] = &[
    b"cuInit\0",
    b"cuDeviceGet\0",
    b"cuDeviceGetAttribute\0",
    b"cuCtxCreate_v3\0",
    b"cuCtxDestroy_v2\0",
    b"cuCtxSetCurrent\0",
    b"cuCtxGetCurrent\0",
    b"cuCtxSynchronize\0",
    b"cuGetErrorName\0",
    b"cuGetErrorString\0",
    b"cuStreamCreate\0",
    b"cuStreamSynchronize\0",
    b"cuStreamDestroy_v2\0",
    b"cuStreamBeginCapture_v2\0",
    b"cuStreamEndCapture\0",
    b"cuMemAllocAsync\0",
    b"cuMemFreeAsync\0",
    b"cuMemAlloc_v2\0",
    b"cuMemFree_v2\0",
    b"cuMemHostAlloc\0",
    b"cuMemFreeHost\0",
    b"cuMemsetD8Async\0",
    b"cuMemcpyHtoDAsync_v2\0",
    b"cuMemcpyDtoHAsync_v2\0",
    b"cuEventCreate\0",
    b"cuEventSynchronize\0",
    b"cuEventDestroy_v2\0",
    b"cuModuleLoadData\0",
    b"cuModuleGetFunction\0",
    b"cuModuleUnload\0",
    b"cuLaunchKernel\0",
    b"cuGraphInstantiateWithFlags\0",
    b"cuGraphExecDestroy\0",
    b"cuGraphDestroy\0",
    b"cuGraphLaunch\0",
    b"cuGraphUpload\0",
];

fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    // SAFETY: callers give each native export its documented C ABI type.
    unsafe { library.get::<T>(name).map(|value| *value) }.map_err(|error| {
        backend(format!(
            "CUDA runtime missing symbol {}: {error}",
            String::from_utf8_lossy(&name[..name.len() - 1])
        ))
    })
}

fn check_driver(library: &Library) -> Result<()> {
    let version: unsafe extern "C" fn(*mut c_int) -> c_int =
        symbol(library, b"cuDriverGetVersion\0")?;
    let mut installed = 0;
    let code = unsafe { version(&mut installed) };
    if code != 0 {
        return Err(backend(format!(
            "CUDA cuDriverGetVersion failed with status {code}"
        )));
    }
    if installed < 12000 {
        return Err(backend(format!(
            "CUDA DFN3 requires driver API 12.0+; installed API version is {installed}"
        )));
    }
    for name in DRIVER_SYMBOLS {
        // Only presence is inspected here; these addresses are never invoked
        // with this erased type. Cudarc loads the correctly typed functions.
        let _: unsafe extern "C" fn() = symbol(library, name)?;
    }
    Ok(())
}

pub(crate) fn driver() -> Result<Library> {
    // Match cudarc's exact ordering so the library checked here is the library
    // its global loader opens. Retain this handle through Gpu destruction.
    let names = ["cuda", "nvcuda"]
        .into_iter()
        .flat_map(cudarc::get_lib_name_candidates);
    for name in names {
        if let Ok(library) = unsafe { Library::new(&name) } {
            check_driver(&library)?;
            return Ok(library);
        }
    }
    Err(backend(
        "CUDA driver library not found; install an NVIDIA CUDA 12.0+ driver (nvcuda.dll / libcuda.so.1)",
    ))
}

type ProgramHandle = *mut c_void;
type Version = unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int;
type Create = unsafe extern "C" fn(
    *mut ProgramHandle,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> c_int;
type Compile = unsafe extern "C" fn(ProgramHandle, c_int, *const *const c_char) -> c_int;
type Size = unsafe extern "C" fn(ProgramHandle, *mut usize) -> c_int;
type Get = unsafe extern "C" fn(ProgramHandle, *mut c_char) -> c_int;
type Destroy = unsafe extern "C" fn(*mut ProgramHandle) -> c_int;
type ErrorString = unsafe extern "C" fn(c_int) -> *const c_char;

pub(crate) struct Nvrtc {
    create: Create,
    compile: Compile,
    log_size: Size,
    log: Get,
    ptx_size: Size,
    ptx: Get,
    destroy: Destroy,
    error_string: ErrorString,
    _library: Library,
}

impl Nvrtc {
    pub(crate) fn load() -> Result<Self> {
        // CUDA >=12 uses one stable filename per major, including Windows' 0
        // in nvrtc64_120_0.dll. Cudarc's single-version candidates do not cover
        // both official major-version names, so load this small C ABI directly.
        #[cfg(target_os = "windows")]
        const NAMES: &[&str] = &["nvrtc64_130_0.dll", "nvrtc64_120_0.dll"];
        #[cfg(target_os = "linux")]
        const NAMES: &[&str] = &["libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so"];
        let mut failures = Vec::new();
        for name in NAMES {
            match Self::open(name) {
                Ok(runtime) => return Ok(runtime),
                Err(error) => failures.push(format!("{name}: {error}")),
            }
        }
        Err(backend(format!(
            "CUDA NVRTC 12/13 unavailable; add its bin directory to PATH (Windows) or library directory to LD_LIBRARY_PATH (Linux). {}",
            failures.join("; ")
        )))
    }

    fn open(name: &str) -> Result<Self> {
        let library = unsafe { Library::new(name) }.map_err(anr_core::Error::backend)?;
        let version: Version = symbol(&library, b"nvrtcVersion\0")?;
        let mut major = 0;
        let mut minor = 0;
        let code = unsafe { version(&mut major, &mut minor) };
        if code != 0 || !(12..=13).contains(&major) {
            return Err(backend(format!(
                "NVRTC 12/13 required; {name} reported {major}.{minor}, status {code}"
            )));
        }
        Ok(Self {
            create: symbol(&library, b"nvrtcCreateProgram\0")?,
            compile: symbol(&library, b"nvrtcCompileProgram\0")?,
            log_size: symbol(&library, b"nvrtcGetProgramLogSize\0")?,
            log: symbol(&library, b"nvrtcGetProgramLog\0")?,
            ptx_size: symbol(&library, b"nvrtcGetPTXSize\0")?,
            ptx: symbol(&library, b"nvrtcGetPTX\0")?,
            destroy: symbol(&library, b"nvrtcDestroyProgram\0")?,
            error_string: symbol(&library, b"nvrtcGetErrorString\0")?,
            _library: library,
        })
    }

    fn check(&self, status: c_int, operation: &str) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        let text = unsafe { (self.error_string)(status) };
        let detail = if text.is_null() {
            "unknown NVRTC error".into()
        } else {
            unsafe { CStr::from_ptr(text) }.to_string_lossy()
        };
        Err(backend(format!(
            "CUDA NVRTC {operation}: {detail} ({status})"
        )))
    }

    pub(crate) fn compile(&self, source: &str, major: i32, minor: i32) -> Result<Ptx> {
        let source = CString::new(source).map_err(anr_core::Error::backend)?;
        let mut program = Program {
            handle: std::ptr::null_mut(),
            runtime: self,
        };
        self.check(
            unsafe {
                (self.create)(
                    &mut program.handle,
                    source.as_ptr(),
                    c"dfn3.cu".as_ptr(),
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                )
            },
            "create program",
        )?;
        let architecture = CString::new(format!("--gpu-architecture=compute_{major}{minor}"))
            .map_err(anr_core::Error::backend)?;
        let options = [
            c"--fmad=false".as_ptr(),
            c"--ftz=false".as_ptr(),
            c"--prec-sqrt=true".as_ptr(),
            c"--prec-div=true".as_ptr(),
            c"--std=c++11".as_ptr(),
            architecture.as_ptr(),
        ];
        let status =
            unsafe { (self.compile)(program.handle, options.len() as c_int, options.as_ptr()) };
        if status != 0 {
            let log = self.bytes(program.handle, self.log_size, self.log, "compilation log")?;
            return Err(backend(format!(
                "CUDA NVRTC compilation failed ({status}): {}",
                String::from_utf8_lossy(&log)
            )));
        }
        let bytes = self.bytes(program.handle, self.ptx_size, self.ptx, "PTX")?;
        let text = CStr::from_bytes_until_nul(&bytes)
            .map_err(anr_core::Error::backend)?
            .to_str()
            .map_err(anr_core::Error::backend)?;
        let ptx = Ptx::from_src(text);
        let status = unsafe { (self.destroy)(&mut program.handle) };
        self.check(status, "destroy program")?;
        Ok(ptx)
    }

    fn bytes(
        &self,
        program: ProgramHandle,
        size: Size,
        get: Get,
        operation: &str,
    ) -> Result<Vec<u8>> {
        let mut len = 0;
        self.check(unsafe { size(program, &mut len) }, operation)?;
        let mut bytes = vec![0u8; len];
        self.check(
            unsafe { get(program, bytes.as_mut_ptr().cast()) },
            operation,
        )?;
        Ok(bytes)
    }
}

struct Program<'a> {
    handle: ProgramHandle,
    runtime: &'a Nvrtc,
}
impl Drop for Program<'_> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let _ = unsafe { (self.runtime.destroy)(&mut self.handle) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_real_library_without_cuda_exports_returns_an_error() {
        #[cfg(target_os = "windows")]
        let library = unsafe { Library::new("kernel32.dll") }.unwrap();
        #[cfg(target_os = "linux")]
        let library: Library = libloading::os::unix::Library::this().into();
        assert!(check_driver(&library).is_err());
    }

    #[cfg(target_os = "windows")]
    #[test]
    #[ignore = "requires an official CUDA 12 NVRTC installation on PATH"]
    fn official_nvrtc12_dll_compiles_the_actual_dfn3_kernels() {
        let runtime = Nvrtc::open("nvrtc64_120_0.dll").unwrap();
        runtime.compile(anr_dfn3::shaders::cuda(), 7, 5).unwrap();
    }
}
