use anr_core::Result;
use anr_dfn3::{
    Dispatch, Executor, FRAME_SIZE, Model, PARAMETER_BYTES, STATE_SIZE, backend, shaders,
};
use std::ffi::{CString, c_void};
use std::mem::ManuallyDrop;
use std::ptr;
use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_FAILED, WAIT_OBJECT_0, WAIT_TIMEOUT};
use windows::Win32::Graphics::Direct3D::Fxc::{
    D3DCOMPILE_ENABLE_STRICTNESS, D3DCOMPILE_IEEE_STRICTNESS, D3DCOMPILE_OPTIMIZATION_LEVEL3,
    D3DCompile,
};
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_11_0, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_ERROR_NOT_FOUND, IDXGIFactory1,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{Interface, PCSTR, PCWSTR, s};

const PARAMETER_STRIDE: u32 = 256;
const FRAME_BYTES: u64 = (FRAME_SIZE * size_of::<f32>()) as u64;
const STATE_BYTES: u64 = (STATE_SIZE * size_of::<f32>()) as u64;

fn native<T>(result: windows::core::Result<T>) -> Result<T> {
    result.map_err(|error| backend(format!("D3D12: {error}")))
}

struct Event(HANDLE);
impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.0);
        }
    }
}

/// A resource and, for host heaps, its lifetime-long mapping. Callers only
/// access mapped memory after the previous submission has completed.
struct Buffer {
    resource: ID3D12Resource,
    mapping: *mut c_void,
    size: u64,
}
impl Buffer {
    fn new(
        device: &ID3D12Device,
        size: u64,
        heap: D3D12_HEAP_TYPE,
        state: D3D12_RESOURCE_STATES,
        flags: D3D12_RESOURCE_FLAGS,
    ) -> Result<Self> {
        let properties = D3D12_HEAP_PROPERTIES {
            Type: heap,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
            ..Default::default()
        };
        let description = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut resource = None;
        unsafe {
            native(device.CreateCommittedResource(
                &properties,
                D3D12_HEAP_FLAG_NONE,
                &description,
                state,
                None,
                &mut resource,
            ))?;
        }
        let resource = resource.ok_or_else(|| backend("D3D12 returned no buffer"))?;
        let mut buffer = Self {
            resource,
            mapping: ptr::null_mut(),
            size,
        };
        if heap != D3D12_HEAP_TYPE_DEFAULT {
            let range = D3D12_RANGE {
                Begin: 0,
                End: if heap == D3D12_HEAP_TYPE_READBACK {
                    size as usize
                } else {
                    0
                },
            };
            unsafe {
                native(
                    buffer
                        .resource
                        .Map(0, Some(&range), Some(&mut buffer.mapping)),
                )?;
            }
            if buffer.mapping.is_null() {
                return Err(backend("D3D12 returned a null buffer mapping"));
            }
        }
        Ok(buffer)
    }
    fn device(device: &ID3D12Device, size: u64, writable: bool) -> Result<Self> {
        Self::new(
            device,
            size,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COPY_DEST,
            if writable {
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            } else {
                D3D12_RESOURCE_FLAG_NONE
            },
        )
    }
    fn upload(device: &ID3D12Device, size: u64) -> Result<Self> {
        Self::new(
            device,
            size,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            D3D12_RESOURCE_FLAG_NONE,
        )
    }
    fn address(&self) -> u64 {
        unsafe { self.resource.GetGPUVirtualAddress() }
    }
    // Offsets are fixed by the hash-locked model; no upload writes race a GPU read.
    fn write(&self, offset: usize, bytes: &[u8]) {
        debug_assert!(!self.mapping.is_null() && offset + bytes.len() <= self.size as usize);
        unsafe {
            ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.mapping.cast::<u8>().add(offset),
                bytes.len(),
            );
        }
    }
    fn zero(&self) {
        debug_assert!(!self.mapping.is_null());
        unsafe {
            ptr::write_bytes(self.mapping.cast::<u8>(), 0, self.size as usize);
        }
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        if !self.mapping.is_null() {
            unsafe {
                self.resource.Unmap(0, None);
            }
        }
    }
}

struct Context {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    submitted_list: [Option<ID3D12CommandList>; 1],
    fence: ID3D12Fence,
    event: Event,
    next_value: u64,
    pending: Option<u64>,
    signaled: bool,
}
impl Context {
    fn new() -> Result<Self> {
        unsafe {
            let factory: IDXGIFactory1 = native(CreateDXGIFactory1())?;
            let mut selected = None;
            let mut index = 0;
            loop {
                let adapter = match factory.EnumAdapters1(index) {
                    Ok(adapter) => adapter,
                    Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => break,
                    Err(error) => {
                        return Err(backend(format!("DXGI adapter enumeration: {error}")));
                    }
                };
                index += 1;
                let description = native(adapter.GetDesc1())?;
                if description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                    continue;
                }
                let mut device: Option<ID3D12Device> = None;
                if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device).is_ok()
                    && let Some(device) = device
                {
                    selected = Some(device);
                    break;
                }
            }
            let device = selected.ok_or_else(|| {
                backend("no hardware adapter supports D3D12 compute; WARP is not used")
            })?;
            let queue: ID3D12CommandQueue =
                native(device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
                    ..Default::default()
                }))?;
            let allocator: ID3D12CommandAllocator =
                native(device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE))?;
            let list: ID3D12GraphicsCommandList = native(device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_COMPUTE,
                &allocator,
                None,
            ))?;
            native(list.Close())?;
            let submitted_list = [Some(native(list.cast::<ID3D12CommandList>())?)];
            let fence = native(device.CreateFence(0, D3D12_FENCE_FLAG_NONE))?;
            let event = Event(native(CreateEventW(None, false, false, PCWSTR::null()))?);
            Ok(Self {
                device,
                queue,
                allocator,
                list,
                submitted_list,
                fence,
                event,
                next_value: 0,
                pending: None,
                signaled: false,
            })
        }
    }
    fn begin(&self) -> Result<()> {
        debug_assert!(self.pending.is_none());
        unsafe {
            native(self.allocator.Reset())?;
            native(self.list.Reset(&self.allocator, None))
        }
    }
    fn submit(&mut self) -> Result<()> {
        unsafe {
            native(self.list.Close())?;
            self.next_value += 1;
            // Record in-flight ownership before ExecuteCommandLists: even a
            // failed queue Signal must not free any referenced resource.
            self.pending = Some(self.next_value);
            self.signaled = false;
            self.queue.ExecuteCommandLists(&self.submitted_list);
            native(self.queue.Signal(&self.fence, self.next_value))?;
            self.signaled = true;
        }
        Ok(())
    }
    fn completed(&self, value: u64) -> Result<bool> {
        unsafe {
            let completed = self.fence.GetCompletedValue();
            native(self.device.GetDeviceRemovedReason())?;
            if completed == u64::MAX {
                return Err(backend(
                    "D3D12 device was removed while waiting for compute",
                ));
            }
            Ok(completed >= value)
        }
    }
    fn wait(&mut self) -> Result<()> {
        let Some(value) = self.pending else {
            return Ok(());
        };
        if !self.signaled {
            return Err(backend(
                "D3D12 queue did not signal its in-flight submission",
            ));
        }
        if !self.completed(value)? {
            unsafe {
                native(self.fence.SetEventOnCompletion(value, self.event.0))?;
            }
            loop {
                let status = unsafe { WaitForSingleObject(self.event.0, 1000) };
                if status == WAIT_FAILED {
                    return Err(backend(format!(
                        "D3D12 fence wait failed: {}",
                        windows::core::Error::from_thread()
                    )));
                }
                if status != WAIT_OBJECT_0 && status != WAIT_TIMEOUT {
                    return Err(backend(format!(
                        "D3D12 fence wait returned unexpected status {}",
                        status.0
                    )));
                }
                if self.completed(value)? {
                    break;
                }
            }
        }
        self.pending = None;
        Ok(())
    }
    fn drain(&mut self) {
        let Some(value) = self.pending else {
            return;
        };
        if self.wait().is_ok() {
            return;
        }
        // A broken event or failed SetEventOnCompletion does not cancel GPU
        // work. Poll the same fence before releasing buffers, mappings, or the
        // event. A removed device cannot execute further commands.
        loop {
            unsafe {
                if self.device.GetDeviceRemovedReason().is_err() {
                    break;
                }
                if !self.signaled && self.queue.Signal(&self.fence, value).is_ok() {
                    self.signaled = true;
                }
                let completed = self.fence.GetCompletedValue();
                if completed == u64::MAX || (self.signaled && completed >= value) {
                    break;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        self.pending = None;
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        self.drain();
    }
}

fn transition(
    list: &ID3D12GraphicsCommandList,
    buffer: &Buffer,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    let mut barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(buffer.resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    unsafe {
        list.ResourceBarrier(std::slice::from_ref(&barrier));
        // Windows unions deliberately do not release their COM fields.
        ManuallyDrop::drop(&mut (*barrier.Anonymous.Transition).pResource);
    }
}

fn uav_barrier(list: &ID3D12GraphicsCommandList) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: ManuallyDrop::new(None),
            }),
        },
    };
    unsafe {
        list.ResourceBarrier(&[barrier]);
    }
}

fn blob_bytes(blob: &ID3DBlob) -> &[u8] {
    unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()) }
}

fn root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature> {
    let types = [
        D3D12_ROOT_PARAMETER_TYPE_SRV,
        D3D12_ROOT_PARAMETER_TYPE_SRV,
        D3D12_ROOT_PARAMETER_TYPE_UAV,
        D3D12_ROOT_PARAMETER_TYPE_UAV,
        D3D12_ROOT_PARAMETER_TYPE_CBV,
    ];
    let parameters = std::array::from_fn::<_, 5, _>(|index| D3D12_ROOT_PARAMETER {
        ParameterType: types[index],
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR {
                ShaderRegister: index as u32,
                RegisterSpace: 0,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    });
    let description = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: parameters.len() as u32,
        pParameters: parameters.as_ptr(),
        ..Default::default()
    };
    let mut blob = None;
    let mut diagnostic = None;
    unsafe {
        if let Err(error) = D3D12SerializeRootSignature(
            &description,
            D3D_ROOT_SIGNATURE_VERSION_1,
            &mut blob,
            Some(&mut diagnostic),
        ) {
            let message = diagnostic
                .as_ref()
                .map(|blob| String::from_utf8_lossy(blob_bytes(blob)))
                .unwrap_or_default();
            return Err(backend(format!("D3D12 root signature: {error}: {message}")));
        }
        let blob = blob.ok_or_else(|| backend("D3D12 returned no root signature bytecode"))?;
        native(device.CreateRootSignature(0, blob_bytes(&blob)))
    }
}

fn pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    dispatch: &Dispatch,
) -> Result<ID3D12PipelineState> {
    let hlsl = shaders::hlsl(dispatch)?;
    let entry = CString::new(hlsl.entry_point)
        .map_err(|_| backend("HLSL entry point contains a null byte"))?;
    let mut bytecode = None;
    let mut diagnostic = None;
    unsafe {
        if let Err(error) = D3DCompile(
            hlsl.source.as_ptr().cast(),
            hlsl.source.len(),
            PCSTR::null(),
            None,
            None,
            PCSTR(entry.as_ptr().cast()),
            s!("cs_5_1"),
            D3DCOMPILE_ENABLE_STRICTNESS
                | D3DCOMPILE_IEEE_STRICTNESS
                | D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0,
            &mut bytecode,
            Some(&mut diagnostic),
        ) {
            let message = diagnostic
                .as_ref()
                .map(|blob| String::from_utf8_lossy(blob_bytes(blob)))
                .unwrap_or_default();
            return Err(backend(format!(
                "D3DCompile {dispatch:?}: {error}: {message}"
            )));
        }
        let bytecode = bytecode.ok_or_else(|| backend("D3DCompile returned no shader bytecode"))?;
        let mut description = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(root.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytecode.GetBufferPointer(),
                BytecodeLength: bytecode.GetBufferSize(),
            },
            ..Default::default()
        };
        let result = device.CreateComputePipelineState(&description);
        ManuallyDrop::drop(&mut description.pRootSignature);
        native(result)
    }
}

pub(crate) struct Gpu {
    context: Context,
    model: Model,
    root: ID3D12RootSignature,
    pipelines: Vec<ID3D12PipelineState>,
    weights: Buffer,
    states: [Buffer; 2],
    scratch: Buffer,
    parameters: Buffer,
    input: Buffer,
    readback: Buffer,
    zeros: Buffer,
    initial_upload: Option<Buffer>,
    state_in: usize,
    poisoned: bool,
}

impl Gpu {
    fn initialize(&mut self) -> Result<()> {
        self.context.begin()?;
        let list = &self.context.list;
        let upload = self
            .initial_upload
            .as_ref()
            .ok_or_else(|| backend("missing D3D12 weight upload"))?;
        unsafe {
            list.CopyBufferRegion(
                &self.weights.resource,
                0,
                &upload.resource,
                0,
                self.weights.size,
            );
            for state in &self.states {
                list.CopyBufferRegion(&state.resource, 0, &self.zeros.resource, 0, STATE_BYTES);
            }
            list.CopyBufferRegion(
                &self.scratch.resource,
                0,
                &self.zeros.resource,
                0,
                self.scratch.size,
            );
        }
        transition(
            list,
            &self.weights,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
        transition(
            list,
            &self.states[0],
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
        transition(
            list,
            &self.states[1],
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        transition(
            list,
            &self.scratch,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        self.context.submit()?;
        self.context.wait()?;
        self.initial_upload = None;
        self.model.weights = Vec::new();
        Ok(())
    }

    fn execute(
        &mut self,
        input: &[f32; FRAME_SIZE],
        output: &mut [f32; FRAME_SIZE],
        attenuation: f32,
        reset: bool,
    ) -> Result<()> {
        self.context.wait()?;
        if self.model.update_attenuation(attenuation) {
            let index = self.model.attenuation_dispatch;
            let parameters = &self.model.dispatches[index].params;
            // Only the mutable attenuation words are changed, after the prior
            // frame fence. All CBVs retain their fixed 256-byte-aligned slice.
            self.parameters.write(
                index * PARAMETER_STRIDE as usize + 22 * 4,
                &parameters[22].to_le_bytes(),
            );
            self.parameters.write(
                index * PARAMETER_STRIDE as usize + 23 * 4,
                &parameters[23].to_le_bytes(),
            );
        }
        let input_bytes = unsafe {
            std::slice::from_raw_parts(input.as_ptr().cast::<u8>(), FRAME_BYTES as usize)
        };
        self.input.write(0, input_bytes);
        self.context.begin()?;
        let list = &self.context.list;
        let state_in = &self.states[self.state_in];
        let state_out = &self.states[1 - self.state_in];
        if reset {
            transition(
                list,
                state_in,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            transition(
                list,
                state_out,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            unsafe {
                list.CopyBufferRegion(&state_in.resource, 0, &self.zeros.resource, 0, STATE_BYTES);
                list.CopyBufferRegion(&state_out.resource, 0, &self.zeros.resource, 0, STATE_BYTES);
            }
            transition(
                list,
                state_in,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            );
            transition(
                list,
                state_out,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            );
        }
        transition(
            list,
            &self.scratch,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        unsafe {
            if reset {
                list.CopyBufferRegion(
                    &self.scratch.resource,
                    0,
                    &self.zeros.resource,
                    0,
                    self.scratch.size,
                );
            }
            list.CopyBufferRegion(
                &self.scratch.resource,
                u64::from(self.model.input) * 4,
                &self.input.resource,
                0,
                FRAME_BYTES,
            );
        }
        transition(
            list,
            &self.scratch,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        unsafe {
            list.SetComputeRootSignature(&self.root);
            list.SetComputeRootShaderResourceView(0, self.weights.address());
            list.SetComputeRootShaderResourceView(1, state_in.address());
            list.SetComputeRootUnorderedAccessView(2, state_out.address());
            list.SetComputeRootUnorderedAccessView(3, self.scratch.address());
            for (index, dispatch) in self.model.dispatches.iter().enumerate() {
                list.SetPipelineState(&self.pipelines[index]);
                list.SetComputeRootConstantBufferView(
                    4,
                    self.parameters.address() + index as u64 * u64::from(PARAMETER_STRIDE),
                );
                list.Dispatch(dispatch.groups, 1, 1);
                // Scratch/state producers must be visible to the next dispatch.
                uav_barrier(list);
            }
        }
        transition(
            list,
            &self.scratch,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        unsafe {
            list.CopyBufferRegion(
                &self.readback.resource,
                0,
                &self.scratch.resource,
                u64::from(self.model.output) * 4,
                FRAME_BYTES,
            );
        }
        transition(
            list,
            &self.scratch,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        // The output becomes the next frame's SRV; the old input becomes UAV.
        transition(
            list,
            state_in,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        transition(
            list,
            state_out,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
        self.context.submit()?;
        self.context.wait()?;
        unsafe {
            ptr::copy_nonoverlapping(
                self.readback.mapping.cast::<f32>(),
                output.as_mut_ptr(),
                FRAME_SIZE,
            );
        }
        self.state_in = 1 - self.state_in;
        Ok(())
    }
}

impl Gpu {
    pub(crate) fn new(model: Model) -> Result<Self> {
        let context = Context::new()?;
        let device = &context.device;
        let root = root_signature(device)?;
        let mut pipelines = Vec::with_capacity(model.dispatches.len());
        for dispatch in &model.dispatches {
            pipelines.push(pipeline(device, &root, dispatch)?);
        }
        let weight_bytes = (model.weights.len() * size_of::<f32>()) as u64;
        let scratch_bytes = u64::from(model.scratch_len) * 4;
        let weights = Buffer::device(device, weight_bytes, false)?;
        let states = [
            Buffer::device(device, STATE_BYTES, true)?,
            Buffer::device(device, STATE_BYTES, true)?,
        ];
        let scratch = Buffer::device(device, scratch_bytes, true)?;
        debug_assert_eq!(PARAMETER_BYTES, 128);
        let parameter_bytes = model.parameter_bytes(PARAMETER_STRIDE);
        let parameters = Buffer::upload(device, parameter_bytes.len() as u64)?;
        parameters.write(0, &parameter_bytes);
        let input = Buffer::upload(device, FRAME_BYTES)?;
        let readback = Buffer::new(
            device,
            FRAME_BYTES,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_FLAG_NONE,
        )?;
        let zeros = Buffer::upload(device, STATE_BYTES.max(scratch_bytes))?;
        zeros.zero();
        let initial_upload = Buffer::upload(device, weight_bytes)?;
        let bytes = unsafe {
            std::slice::from_raw_parts(model.weights.as_ptr().cast::<u8>(), weight_bytes as usize)
        };
        initial_upload.write(0, bytes);
        // No work is submitted until every resource has an owning Gpu. Any
        // constructor error above is ordinary COM/mapping/event RAII cleanup.
        let mut gpu = Self {
            context,
            model,
            root,
            pipelines,
            weights,
            states,
            scratch,
            parameters,
            input,
            readback,
            zeros,
            initial_upload: Some(initial_upload),
            state_in: 0,
            poisoned: false,
        };
        gpu.initialize()?;
        Ok(gpu)
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
        if self.poisoned {
            return Err(backend(
                "D3D12 execution failed; recreate the denoiser before reuse",
            ));
        }
        let result = self.execute(input, output, attenuation, reset);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
}
impl Drop for Gpu {
    fn drop(&mut self) {
        // Runs before field destruction, including when initial upload fails.
        self.context.drain();
    }
}
