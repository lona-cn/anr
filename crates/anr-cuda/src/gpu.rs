use crate::runtime;
use anr_core::Result;
use anr_dfn3::{Executor, FRAME_SIZE, Model, STATE_SIZE, ShaderKind, backend, shaders};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, DevicePtr, PinnedHostSlice,
    result, sys,
};
use std::ffi::c_void;
use std::sync::Arc;

fn cuda(error: impl std::fmt::Display) -> anr_core::Error {
    backend(format!("CUDA DFN3: {error}"))
}

// Drop order is intentional: graphs first, then their captured host/device
// allocations and module, finally stream/context. Drop explicitly drains first.
pub(crate) struct Gpu {
    graphs: Vec<CudaGraph>,
    host_input: PinnedHostSlice<f32>,
    host_output: PinnedHostSlice<f32>,
    host_attenuation: PinnedHostSlice<u32>,
    _weights: CudaSlice<f32>,
    _states: [CudaSlice<f32>; 2],
    _scratch: CudaSlice<f32>,
    _parameters: CudaSlice<u32>,
    functions: [CudaFunction; 3],
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
    model: Model,
    weights_ptr: sys::CUdeviceptr,
    state_ptrs: [sys::CUdeviceptr; 2],
    scratch_ptr: sys::CUdeviceptr,
    parameters_ptr: sys::CUdeviceptr,
    input_ptr: *mut f32,
    output_ptr: *mut f32,
    parity: usize,
    failed: bool,
    _driver: libloading::Library,
}

impl Gpu {
    pub(crate) fn new(mut model: Model) -> Result<Self> {
        // Resolve driver version and every API used by cudarc before entering
        // its panic-on-missing-symbol loader. This also handles panic=abort.
        let driver = runtime::driver()?;
        let compiler = runtime::Nvrtc::load()?;
        // An independent context avoids modifying another CUDA user's primary
        // context event-tracking policy or stream state.
        let context = CudaContext::new_non_primary(0, 0).map_err(cuda)?;
        // All operations belong to this one worker-local stream. Raw captured
        // pointers are owned below; every host access and Drop is synchronized.
        unsafe { context.disable_event_tracking() };
        let stream = context.new_stream().map_err(cuda)?;
        let major = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(cuda)?;
        let minor = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(cuda)?;
        let ptx = compiler.compile(shaders::cuda(), major, minor)?;
        let module = context.load_module(ptx).map_err(|error| backend(format!("CUDA DFN3 loading compiled PTX failed: {error}; the driver must support the installed NVRTC's generated PTX version")))?;
        let functions = [
            module.load_function("network").map_err(cuda)?,
            module.load_function("dsp").map_err(cuda)?,
            module.load_function("fft").map_err(cuda)?,
        ];
        let weights = stream.clone_htod(&model.weights).map_err(cuda)?;
        let states = [
            stream.alloc_zeros::<f32>(STATE_SIZE).map_err(cuda)?,
            stream.alloc_zeros::<f32>(STATE_SIZE).map_err(cuda)?,
        ];
        let scratch = stream
            .alloc_zeros::<f32>(model.scratch_len as usize)
            .map_err(cuda)?;
        let packed: Vec<u32> = model
            .dispatches
            .iter()
            .flat_map(|dispatch| dispatch.params)
            .collect();
        let parameters = stream.clone_htod(&packed).map_err(cuda)?;
        let mut host_input =
            unsafe { context.alloc_pinned_with_flags::<f32>(FRAME_SIZE, 0) }.map_err(cuda)?;
        let mut host_output =
            unsafe { context.alloc_pinned_with_flags::<f32>(FRAME_SIZE, 0) }.map_err(cuda)?;
        let mut host_attenuation =
            unsafe { context.alloc_pinned_with_flags::<u32>(2, 0) }.map_err(cuda)?;
        host_attenuation.as_mut_slice().map_err(cuda)?.fill(0);
        host_input.as_mut_slice().map_err(cuda)?.fill(0.0);
        host_output.as_mut_slice().map_err(cuda)?.fill(0.0);
        stream.synchronize().map_err(cuda)?;
        // No CPU weight copy is retained after the one-time upload completes.
        model.weights = Vec::new();
        let weights_ptr = weights.device_ptr(&stream).0;
        let state_ptrs = [
            states[0].device_ptr(&stream).0,
            states[1].device_ptr(&stream).0,
        ];
        let scratch_ptr = scratch.device_ptr(&stream).0;
        let parameters_ptr = parameters.device_ptr(&stream).0;
        let input_ptr = host_input.as_mut_ptr().map_err(cuda)?;
        let output_ptr = host_output.as_mut_ptr().map_err(cuda)?;
        let mut gpu = Self {
            graphs: Vec::with_capacity(2),
            host_input,
            host_output,
            host_attenuation,
            _weights: weights,
            _states: states,
            _scratch: scratch,
            _parameters: parameters,
            functions,
            stream,
            _context: context,
            model,
            weights_ptr,
            state_ptrs,
            scratch_ptr,
            parameters_ptr,
            input_ptr,
            output_ptr,
            parity: 0,
            failed: false,
            _driver: driver,
        };
        for parity in 0..2 {
            let graph = gpu.capture(parity)?;
            gpu.graphs.push(graph);
            gpu.graphs[parity].upload().map_err(cuda)?;
        }
        gpu.stream.synchronize().map_err(cuda)?;
        Ok(gpu)
    }

    fn capture(&self, parity: usize) -> Result<CudaGraph> {
        self.stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(cuda)?;
        let captured = self.capture_operations(parity);
        // End even an invalidated capture before any allocation can be freed.
        let graph = self.stream.end_capture(sys::CUgraphInstantiate_flags(0));
        captured?;
        graph
            .map_err(cuda)?
            .ok_or_else(|| backend("CUDA DFN3 produced an empty frame graph"))
    }

    fn capture_operations(&self, parity: usize) -> Result<()> {
        let stream = self.stream.cu_stream();
        // SAFETY: these pinned addresses and all device buffers remain stable
        // until graphs are destroyed; the private stream serializes all uses.
        unsafe {
            result::memcpy_htod_async(
                self.scratch_ptr + u64::from(self.model.input) * 4,
                std::slice::from_raw_parts(self.input_ptr, FRAME_SIZE),
                stream,
            )
            .map_err(cuda)?;
        }
        for (index, dispatch) in self.model.dispatches.iter().enumerate() {
            let mut weights = self.weights_ptr;
            let mut state_in = self.state_ptrs[parity];
            let mut state_out = self.state_ptrs[1 - parity];
            let mut scratch = self.scratch_ptr;
            let mut parameters = self.parameters_ptr + index as u64 * 128;
            let mut args: [*mut c_void; 5] = [
                (&mut weights as *mut sys::CUdeviceptr).cast(),
                (&mut state_in as *mut sys::CUdeviceptr).cast(),
                (&mut state_out as *mut sys::CUdeviceptr).cast(),
                (&mut scratch as *mut sys::CUdeviceptr).cast(),
                (&mut parameters as *mut sys::CUdeviceptr).cast(),
            ];
            let block = if dispatch.shader == ShaderKind::Network {
                64
            } else {
                128
            };
            // CUDA copies argument values into the capture node here; the local
            // argument array need not outlive this call. Pointed-to buffers do.
            unsafe {
                result::launch_kernel(
                    self.functions[dispatch.shader.index()].cu_function(),
                    (dispatch.groups, 1, 1),
                    (block, 1, 1),
                    0,
                    stream,
                    &mut args,
                )
                .map_err(cuda)?;
            }
        }
        unsafe {
            result::memcpy_dtoh_async(
                std::slice::from_raw_parts_mut(self.output_ptr, FRAME_SIZE),
                self.scratch_ptr + u64::from(self.model.output) * 4,
                stream,
            )
            .map_err(cuda)?;
        }
        Ok(())
    }

    fn execute(
        &mut self,
        input: &[f32; FRAME_SIZE],
        output: &mut [f32; FRAME_SIZE],
        attenuation: f32,
        reset: bool,
    ) -> Result<()> {
        if reset {
            // Both graph parities must see zero state after a reset.
            for pointer in self.state_ptrs {
                unsafe {
                    result::memset_d8_async(pointer, 0, STATE_SIZE * 4, self.stream.cu_stream())
                }
                .map_err(cuda)?;
            }
            unsafe {
                result::memset_d8_async(
                    self.scratch_ptr,
                    0,
                    self.model.scratch_len as usize * 4,
                    self.stream.cu_stream(),
                )
            }
            .map_err(cuda)?;
            self.parity = 0;
        }
        if self.model.update_attenuation(attenuation) {
            // Fixed graph parameter addresses never change. Only the two words
            // read by spectral filtering change, ordered before graph replay.
            let dispatch = self.model.attenuation_dispatch;
            let words = &self.model.dispatches[dispatch].params[22..24];
            // Persistent pinned storage remains valid through frame completion.
            // The copy and graph replay use the SAME non-blocking stream.
            self.host_attenuation
                .as_mut_slice()
                .map_err(cuda)?
                .copy_from_slice(words);
            unsafe {
                result::memcpy_htod_async(
                    self.parameters_ptr + dispatch as u64 * 128 + 22 * 4,
                    self.host_attenuation.as_slice().map_err(cuda)?,
                    self.stream.cu_stream(),
                )
            }
            .map_err(cuda)?;
        }
        self.host_input
            .as_mut_slice()
            .map_err(cuda)?
            .copy_from_slice(input);
        self.graphs[self.parity].launch().map_err(cuda)?;
        self.stream.synchronize().map_err(cuda)?;
        output.copy_from_slice(self.host_output.as_slice().map_err(cuda)?);
        self.parity = 1 - self.parity;
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
                "CUDA DFN3 stream is invalid after a GPU error; recreate the denoiser (reset cannot repair device state)",
            ));
        }
        let result = self.execute(input, output, attenuation, reset);
        if result.is_err() {
            self.failed = true;
            let _ = self.stream.synchronize();
        }
        result
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // In-flight graph/memcpy nodes must finish (or report device failure)
        // before graphs, pinned memory, device allocations or modules drop.
        let _ = self.stream.synchronize();
    }
}
