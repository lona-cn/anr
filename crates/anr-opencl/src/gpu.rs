use anr_core::Result;
use anr_dfn3::{Executor, FRAME_SIZE, Model, STATE_SIZE, ShaderKind, backend};
use libloading::Library;
use std::ffi::{c_char, c_void};
use std::mem::{size_of, size_of_val};
use std::ptr::{null, null_mut};

type Handle = *mut c_void;
type ContextCallback =
    Option<unsafe extern "system" fn(*const c_char, *const c_void, usize, *mut c_void)>;
type BuildCallback = Option<unsafe extern "system" fn(Handle, *mut c_void)>;

// Exact OpenCL 1.2 ABI, loaded from the ICD rather than linked to an SDK.
macro_rules! api {
    ($($name:ident: ($($arg:ty),*) -> $ret:ty;)*) => {
        #[allow(non_snake_case)]
        struct Api {
            $($name: unsafe extern "system" fn($($arg),*) -> $ret,)*
            _library: Library,
        }
        impl Api {
            fn load() -> Result<Self> {
                #[cfg(target_os = "windows")]
                const NAMES: &[&str] = &["OpenCL.dll"];
                #[cfg(target_os = "macos")]
                const NAMES: &[&str] = &["/System/Library/Frameworks/OpenCL.framework/OpenCL"];
                #[cfg(not(any(target_os = "windows", target_os = "macos")))]
                const NAMES: &[&str] = &["libOpenCL.so.1", "libOpenCL.so"];
                let mut diagnostics = Vec::new();
                for name in NAMES {
                    // SAFETY: Only the platform OpenCL ICD is loaded; each symbol has
                    // its Khronos OpenCL 1.2 signature and the library is retained.
                    let library = match unsafe { Library::new(name) } {
                        Ok(library) => library,
                        Err(error) => { diagnostics.push(format!("{name}: {error}")); continue; }
                    };
                    unsafe {
                        return Ok(Self {
                            $($name: *library.get(concat!(stringify!($name), "\0").as_bytes())
                                .map_err(|error| backend(format!("OpenCL ICD is missing {}: {error}", stringify!($name))))?,)*
                            _library: library,
                        });
                    }
                }
                Err(backend(format!("OpenCL ICD not found; install your GPU vendor's OpenCL driver (no CPU fallback): {}", diagnostics.join("; "))))
            }
        }
    }
}
api! {
    clGetPlatformIDs: (u32, *mut Handle, *mut u32) -> i32;
    clGetDeviceIDs: (Handle, u64, u32, *mut Handle, *mut u32) -> i32;
    clGetDeviceInfo: (Handle, u32, usize, *mut c_void, *mut usize) -> i32;
    clCreateContext: (*const isize, u32, *const Handle, ContextCallback, *mut c_void, *mut i32) -> Handle;
    clReleaseContext: (Handle) -> i32;
    clCreateCommandQueue: (Handle, Handle, u64, *mut i32) -> Handle;
    clReleaseCommandQueue: (Handle) -> i32;
    clCreateBuffer: (Handle, u64, usize, *mut c_void, *mut i32) -> Handle;
    clReleaseMemObject: (Handle) -> i32;
    clCreateProgramWithSource: (Handle, u32, *const *const c_char, *const usize, *mut i32) -> Handle;
    clBuildProgram: (Handle, u32, *const Handle, *const c_char, BuildCallback, *mut c_void) -> i32;
    clGetProgramBuildInfo: (Handle, Handle, u32, usize, *mut c_void, *mut usize) -> i32;
    clReleaseProgram: (Handle) -> i32;
    clCreateKernel: (Handle, *const c_char, *mut i32) -> Handle;
    clReleaseKernel: (Handle) -> i32;
    clGetKernelWorkGroupInfo: (Handle, Handle, u32, usize, *mut c_void, *mut usize) -> i32;
    clSetKernelArg: (Handle, u32, usize, *const c_void) -> i32;
    clEnqueueWriteBuffer: (Handle, Handle, u32, usize, usize, *const c_void, u32, *const Handle, *mut Handle) -> i32;
    clEnqueueReadBuffer: (Handle, Handle, u32, usize, usize, *mut c_void, u32, *const Handle, *mut Handle) -> i32;
    clEnqueueFillBuffer: (Handle, Handle, *const c_void, usize, usize, usize, u32, *const Handle, *mut Handle) -> i32;
    clEnqueueNDRangeKernel: (Handle, Handle, u32, *const usize, *const usize, *const usize, u32, *const Handle, *mut Handle) -> i32;
    clFinish: (Handle) -> i32;
}

fn check(status: i32, operation: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(backend(format!(
            "OpenCL {operation} failed with status {status}; check the GPU driver and device resources"
        )))
    }
}
fn created(handle: Handle, status: i32, operation: &str) -> Result<Handle> {
    check(status, operation)?;
    if handle.is_null() {
        Err(backend(format!(
            "OpenCL {operation} returned a null handle"
        )))
    } else {
        Ok(handle)
    }
}

impl Api {
    fn device_u32(&self, device: Handle, key: u32) -> Result<u32> {
        let mut value = 0u32;
        // SAFETY: value is writable and exactly the size of this cl_uint property.
        check(
            unsafe {
                (self.clGetDeviceInfo)(
                    device,
                    key,
                    size_of_val(&value),
                    (&mut value as *mut u32).cast(),
                    null_mut(),
                )
            },
            "device info",
        )?;
        Ok(value)
    }
    fn device_u64(&self, device: Handle, key: u32) -> Result<u64> {
        let mut value = 0u64;
        // SAFETY: value is writable and exactly the size of this cl_ulong/bitfield.
        check(
            unsafe {
                (self.clGetDeviceInfo)(
                    device,
                    key,
                    size_of_val(&value),
                    (&mut value as *mut u64).cast(),
                    null_mut(),
                )
            },
            "device info",
        )?;
        Ok(value)
    }
    fn device_size(&self, device: Handle, key: u32) -> Result<usize> {
        let mut value = 0usize;
        // SAFETY: the requested property is size_t.
        check(
            unsafe {
                (self.clGetDeviceInfo)(
                    device,
                    key,
                    size_of_val(&value),
                    (&mut value as *mut usize).cast(),
                    null_mut(),
                )
            },
            "device info",
        )?;
        Ok(value)
    }
    fn device_string(&self, device: Handle, key: u32) -> Result<String> {
        let mut size = 0;
        // SAFETY: first query obtains the required byte count, then a valid buffer.
        unsafe {
            check(
                (self.clGetDeviceInfo)(device, key, 0, null_mut(), &mut size),
                "device string size",
            )?;
            let mut bytes = vec![0u8; size];
            check(
                (self.clGetDeviceInfo)(
                    device,
                    key,
                    bytes.len(),
                    bytes.as_mut_ptr().cast(),
                    null_mut(),
                ),
                "device string",
            )?;
            Ok(String::from_utf8_lossy(&bytes)
                .trim_end_matches('\0')
                .to_owned())
        }
    }
    fn suitable(&self, device: Handle) -> Result<u64> {
        // CL_DEVICE_TYPE_GPU, AVAILABLE, COMPILER_AVAILABLE. Never request ALL/CPU.
        if self.device_u64(device, 0x1000)? & 4 == 0
            || self.device_u32(device, 0x1027)? == 0
            || self.device_u32(device, 0x1028)? == 0
        {
            return Err(backend(
                "requires an available hardware OpenCL GPU with its online compiler",
            ));
        }
        let version = self.device_string(device, 0x103d)?; // OPENCL_C_VERSION
        let supported = version
            .strip_prefix("OpenCL C ")
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.split_once('.'))
            .and_then(|(major, minor)| {
                Some((major.parse::<u32>().ok()?, minor.parse::<u32>().ok()?))
            })
            .is_some_and(|(major, minor)| major > 1 || major == 1 && minor >= 2);
        if !supported {
            return Err(backend(format!(
                "requires OpenCL C 1.2 or later, device reports {version}"
            )));
        }
        let local = self.device_u64(device, 0x1023)?;
        if local < 15_360 || self.device_size(device, 0x1004)? < 128 {
            return Err(backend(
                "GPU requires at least 15360 bytes local memory and 128 work-items per group",
            ));
        }
        let dimensions = self.device_u32(device, 0x1003)? as usize;
        let mut sizes = vec![0usize; dimensions];
        // SAFETY: MAX_WORK_ITEM_SIZES returns dimensions size_t values.
        check(
            unsafe {
                (self.clGetDeviceInfo)(
                    device,
                    0x1005,
                    size_of_val(sizes.as_slice()),
                    sizes.as_mut_ptr().cast(),
                    null_mut(),
                )
            },
            "work-item limits",
        )?;
        if sizes.first().copied().unwrap_or(0) < 128 {
            return Err(backend("GPU requires X work-item limit >=128"));
        }
        // Full FP32 behavior (denormals, INF/NAN, round-to-nearest) and accurate
        // division/sqrt are required; never silently build a fast-math variant.
        let fp = self.device_u64(device, 0x101b)?;
        if fp & (1 | 2 | 4 | 128) != 1 | 2 | 4 | 128 {
            return Err(backend(
                "GPU lacks required FP32 denormals/INF/NaN/round-to-nearest/correctly-rounded divide-sqrt",
            ));
        }
        Ok(local)
    }
    fn choose_device(&self) -> Result<(Handle, u64)> {
        let mut count = 0;
        // SAFETY: count is writable; platform/device vectors match queried counts.
        unsafe {
            check(
                (self.clGetPlatformIDs)(0, null_mut(), &mut count),
                "platform enumeration (install a GPU OpenCL ICD)",
            )?;
            let mut platforms = vec![null_mut(); count as usize];
            if count == 0 {
                return Err(backend(
                    "OpenCL ICD has no platforms; install a GPU vendor runtime",
                ));
            }
            check(
                (self.clGetPlatformIDs)(count, platforms.as_mut_ptr(), null_mut()),
                "platform enumeration",
            )?;
            let mut reasons = Vec::new();
            for platform in platforms {
                let mut count = 0;
                let status = (self.clGetDeviceIDs)(platform, 4, 0, null_mut(), &mut count);
                if status == -1 || count == 0 {
                    continue;
                } // CL_DEVICE_NOT_FOUND
                if status != 0 {
                    reasons.push(format!("GPU enumeration status {status}"));
                    continue;
                }
                let mut devices = vec![null_mut(); count as usize];
                check(
                    (self.clGetDeviceIDs)(platform, 4, count, devices.as_mut_ptr(), null_mut()),
                    "GPU enumeration",
                )?;
                for device in devices {
                    match self.suitable(device) {
                        Ok(local) => return Ok((device, local)),
                        Err(error) => reasons.push(format!(
                            "{}: {error}",
                            self.device_string(device, 0x102b)
                                .unwrap_or_else(|_| "unknown GPU".into())
                        )),
                    }
                }
            }
            Err(backend(format!(
                "No suitable hardware OpenCL 1.2 GPU (CPU fallback is disabled). {}",
                reasons.join("; ")
            )))
        }
    }
}

pub(crate) struct Gpu {
    api: Api,
    context: Handle,
    queue: Handle,
    program: Handle,
    kernels: [Handle; 3],
    // weights, state0, state1, scratch, parameter table
    buffers: [Handle; 5],
    model: Model,
    parity: usize,
    failed: bool,
}
impl Gpu {
    pub(crate) fn new(model: Model) -> Result<Self> {
        let api = Api::load()?;
        let (device, local_limit) = api.choose_device()?;
        // Establish the RAII owner before creating any native object. Every
        // partial-construction error therefore uses exactly the same teardown.
        let mut gpu = Self {
            api,
            context: null_mut(),
            queue: null_mut(),
            program: null_mut(),
            kernels: [null_mut(); 3],
            buffers: [null_mut(); 5],
            model,
            parity: 0,
            failed: false,
        };
        // SAFETY: All handles are checked on creation and owned until Drop;
        // host buffers remain alive through blocking copies and synchronous build.
        unsafe {
            let mut status = 0;
            let context =
                (gpu.api.clCreateContext)(null(), 1, &device, None, null_mut(), &mut status);
            gpu.context = created(context, status, "context creation")?;
            // Zero properties: the OpenCL 1.2 command queue is strictly in-order.
            let queue = (gpu.api.clCreateCommandQueue)(gpu.context, device, 0, &mut status);
            gpu.queue = created(queue, status, "in-order queue creation")?;
            let sizes = [
                size_of_val(gpu.model.weights.as_slice()),
                STATE_SIZE * 4,
                STATE_SIZE * 4,
                gpu.model.scratch_len as usize * 4,
                gpu.model.dispatches.len() * 128,
            ];
            let max_allocation = gpu.api.device_u64(device, 0x1010)?;
            for (i, bytes) in sizes.into_iter().enumerate() {
                if bytes == 0 || bytes as u64 > max_allocation {
                    return Err(backend(format!(
                        "OpenCL buffer {i} requires {bytes} bytes, device allocation limit {max_allocation}"
                    )));
                }
                let flags = if i == 0 || i == 4 { 4 } else { 1 }; // READ_ONLY / READ_WRITE (kernel access)
                let buffer =
                    (gpu.api.clCreateBuffer)(gpu.context, flags, bytes, null_mut(), &mut status);
                gpu.buffers[i] = created(buffer, status, "buffer allocation")?;
            }
            gpu.write(
                gpu.buffers[0],
                0,
                gpu.model.weights.as_ptr().cast(),
                sizes[0],
            )?;
            let parameters = gpu.model.parameter_bytes(128);
            gpu.write(
                gpu.buffers[4],
                0,
                parameters.as_ptr().cast(),
                parameters.len(),
            )?;
            gpu.zero(gpu.buffers[1], sizes[1])?;
            gpu.zero(gpu.buffers[2], sizes[2])?;
            gpu.zero(gpu.buffers[3], sizes[3])?;
            let source = include_str!("kernels.cl");
            let pointer = source.as_ptr().cast::<c_char>();
            let length = source.len();
            let program =
                (gpu.api.clCreateProgramWithSource)(gpu.context, 1, &pointer, &length, &mut status);
            gpu.program = created(program, status, "program creation")?;
            status = (gpu.api.clBuildProgram)(
                gpu.program,
                1,
                &device,
                c"-cl-std=CL1.2 -cl-fp32-correctly-rounded-divide-sqrt".as_ptr(),
                None,
                null_mut(),
            );
            if status != 0 {
                return Err(backend(format!(
                    "OpenCL kernel build failed with status {status}:\n{}",
                    gpu.build_log(device)
                )));
            }
            for (index, name) in [c"network", c"dsp", c"fft"].into_iter().enumerate() {
                let kernel = (gpu.api.clCreateKernel)(gpu.program, name.as_ptr(), &mut status);
                gpu.kernels[index] = created(kernel, status, "kernel creation")?;
                let mut group_limit = 0usize;
                check(
                    (gpu.api.clGetKernelWorkGroupInfo)(
                        kernel,
                        device,
                        0x11b0,
                        size_of_val(&group_limit),
                        (&mut group_limit as *mut usize).cast(),
                        null_mut(),
                    ),
                    "kernel group limit",
                )?;
                let mut local = 0u64;
                check(
                    (gpu.api.clGetKernelWorkGroupInfo)(
                        kernel,
                        device,
                        0x11b2,
                        size_of_val(&local),
                        (&mut local as *mut u64).cast(),
                        null_mut(),
                    ),
                    "kernel local memory",
                )?;
                let required = if index == 0 { 64 } else { 128 };
                if group_limit < required || local > local_limit {
                    return Err(backend(format!(
                        "OpenCL kernel {} cannot launch: group limit {group_limit}, required {required}, local {local}/{local_limit}",
                        name.to_string_lossy()
                    )));
                }
                gpu.arg(kernel, 0, &gpu.buffers[0])?;
                gpu.arg(kernel, 3, &gpu.buffers[3])?;
                gpu.arg(kernel, 4, &gpu.buffers[4])?;
            }
            check((gpu.api.clFinish)(gpu.queue), "initialization completion")?;
        }
        // No host weights are needed after their one-time upload.
        gpu.model.weights = Vec::new();
        Ok(gpu)
    }
    fn build_log(&self, device: Handle) -> String {
        // SAFETY: two-call query supplies a correctly sized writable log buffer.
        unsafe {
            let mut size = 0usize;
            let status = (self.api.clGetProgramBuildInfo)(
                self.program,
                device,
                0x1183,
                0,
                null_mut(),
                &mut size,
            );
            if status != 0 {
                return format!("build log size query failed: {status}");
            }
            let mut bytes = vec![0u8; size];
            let status = (self.api.clGetProgramBuildInfo)(
                self.program,
                device,
                0x1183,
                size,
                bytes.as_mut_ptr().cast(),
                null_mut(),
            );
            if status != 0 {
                return format!("build log query failed: {status}");
            }
            String::from_utf8_lossy(&bytes)
                .trim_end_matches('\0')
                .to_owned()
        }
    }
    fn arg<T>(&self, kernel: Handle, index: u32, value: &T) -> Result<()> {
        // SAFETY: clSetKernelArg copies the argument value synchronously.
        check(
            unsafe {
                (self.api.clSetKernelArg)(kernel, index, size_of::<T>(), (value as *const T).cast())
            },
            "kernel argument",
        )
    }
    unsafe fn write(
        &self,
        buffer: Handle,
        offset: usize,
        source: *const c_void,
        bytes: usize,
    ) -> Result<()> {
        // SAFETY: caller provides bytes readable from source. Blocking write
        // ensures no caller-owned host pointer remains in flight on return.
        check(
            unsafe {
                (self.api.clEnqueueWriteBuffer)(
                    self.queue,
                    buffer,
                    1,
                    offset,
                    bytes,
                    source,
                    0,
                    null(),
                    null_mut(),
                )
            },
            "buffer upload",
        )
    }
    fn zero(&self, buffer: Handle, bytes: usize) -> Result<()> {
        let zero = 0u32;
        // SAFETY: fill copies the four-byte pattern before returning; all buffer
        // sizes are multiples of four, and the queue retains the memory object.
        check(
            unsafe {
                (self.api.clEnqueueFillBuffer)(
                    self.queue,
                    buffer,
                    (&zero as *const u32).cast(),
                    4,
                    0,
                    bytes,
                    0,
                    null(),
                    null_mut(),
                )
            },
            "state/scratch clear",
        )
    }
    fn execute(
        &mut self,
        input: &[f32; FRAME_SIZE],
        output: &mut [f32; FRAME_SIZE],
        attenuation: f32,
        reset: bool,
    ) -> Result<()> {
        if reset {
            self.zero(self.buffers[1], STATE_SIZE * 4)?;
            self.zero(self.buffers[2], STATE_SIZE * 4)?;
            self.parity = 0;
        }
        if self.model.update_attenuation(attenuation) {
            let dispatch = self.model.attenuation_dispatch;
            let words = &self.model.dispatches[dispatch].params[22..24];
            // SAFETY: uploads exactly the two changed words; write is blocking.
            unsafe {
                self.write(
                    self.buffers[4],
                    dispatch * 128 + 22 * 4,
                    words.as_ptr().cast(),
                    8,
                )?;
            }
        }
        // SAFETY: input/output arrays have exactly FRAME_SIZE f32 values; all
        // dispatch handles and offsets come from the validated static plan.
        unsafe {
            self.write(
                self.buffers[3],
                self.model.input as usize * 4,
                input.as_ptr().cast(),
                size_of_val(input),
            )?;
            for kernel in self.kernels {
                self.arg(kernel, 1, &self.buffers[1 + self.parity])?;
                self.arg(kernel, 2, &self.buffers[2 - self.parity])?;
            }
            for (i, dispatch) in self.model.dispatches.iter().enumerate() {
                let kernel = self.kernels[dispatch.shader.index()];
                let offset = (i * 32) as u32;
                self.arg(kernel, 5, &offset)?;
                let local = if dispatch.shader == ShaderKind::Network {
                    64usize
                } else {
                    128usize
                };
                let global = dispatch.groups as usize * local;
                check(
                    (self.api.clEnqueueNDRangeKernel)(
                        self.queue,
                        kernel,
                        1,
                        null(),
                        &global,
                        &local,
                        0,
                        null(),
                        null_mut(),
                    ),
                    "frame kernel enqueue",
                )?;
            }
            // The in-order queue makes the blocking read wait for every prior
            // kernel. No state/scratch readback and no per-frame allocations.
            check(
                (self.api.clEnqueueReadBuffer)(
                    self.queue,
                    self.buffers[3],
                    1,
                    self.model.output as usize * 4,
                    size_of_val(output),
                    output.as_mut_ptr().cast(),
                    0,
                    null(),
                    null_mut(),
                ),
                "output frame completion",
            )?;
        }
        self.parity ^= 1;
        Ok(())
    }
}
impl Executor for Gpu {
    fn frame(
        &mut self,
        input: &[f32; FRAME_SIZE],
        output: &mut [f32; FRAME_SIZE],
        attenuation: f32,
        reset: bool,
    ) -> Result<()> {
        if self.failed {
            return Err(backend(
                "OpenCL stream failed previously; recreate the reducer (no fallback)",
            ));
        }
        let result = self.execute(input, output, attenuation, reset);
        if result.is_err() {
            self.failed = true;
            // SAFETY: drain even after partial enqueue/fatal execution errors.
            // Never let a failed frame reuse ambiguous recurrent state.
            unsafe {
                (self.api.clFinish)(self.queue);
            }
        }
        result
    }
}
impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: null slots represent not-yet-created resources. Drain before
        // releasing kernels/program/buffers/queue/context, including init errors.
        // The dynamic library stays loaded through this entire Drop body.
        unsafe {
            if !self.queue.is_null() {
                (self.api.clFinish)(self.queue);
            }
            for kernel in self.kernels {
                if !kernel.is_null() {
                    (self.api.clReleaseKernel)(kernel);
                }
            }
            if !self.program.is_null() {
                (self.api.clReleaseProgram)(self.program);
            }
            for buffer in self.buffers {
                if !buffer.is_null() {
                    (self.api.clReleaseMemObject)(buffer);
                }
            }
            if !self.queue.is_null() {
                (self.api.clReleaseCommandQueue)(self.queue);
            }
            if !self.context.is_null() {
                (self.api.clReleaseContext)(self.context);
            }
        }
    }
}
