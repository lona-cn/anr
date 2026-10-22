use anr_core::Result;
use anr_dfn3::{FRAME_SIZE, Model, PARAMETER_BYTES, STATE_SIZE, backend};
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

const FRAME_BYTES: u64 = (FRAME_SIZE * size_of::<f32>()) as u64;
const STATE_BYTES: u64 = (STATE_SIZE * size_of::<f32>()) as u64;

// Callbacks only record failures. The worker observes them before publishing
// output; validation, allocation, internal errors, and device loss never panic.
#[derive(Clone, Default)]
pub(super) struct DeviceErrors(Arc<Mutex<Option<String>>>);

impl DeviceErrors {
    fn record(&self, message: String) {
        let mut first = self.0.lock().unwrap_or_else(|error| error.into_inner());
        if first.is_none() {
            *first = Some(message);
        }
    }

    pub(super) fn check(&self) -> Result<()> {
        let first = self.0.lock().unwrap_or_else(|error| error.into_inner());
        match first.as_ref() {
            Some(message) => Err(backend(format!("wgpu: {message}"))),
            None => Ok(()),
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
struct InitializingDevice(Option<wgpu::Device>);

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl Drop for InitializingDevice {
    fn drop(&mut self) {
        if let Some(device) = self.0.take() {
            device.destroy();
        }
    }
}

// GPU resources and the fixed schedule are shared by native and browser waiting.
pub(super) struct Core {
    pub(super) device: wgpu::Device,
    queue: wgpu::Queue,
    states: [wgpu::Buffer; 2],
    scratch: wgpu::Buffer,
    parameters: wgpu::Buffer,
    pub(super) readback: wgpu::Buffer,
    pipelines: Vec<wgpu::ComputePipeline>,
    bind_groups: [wgpu::BindGroup; 2],
    model: Model,
    stride: u32,
    pub(super) current: usize,
    pub(super) errors: DeviceErrors,
}

impl Core {
    pub(super) async fn new(model: Model) -> Result<Self> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            descriptor.backends =
                wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL;
        }
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            descriptor.backends = wgpu::Backends::BROWSER_WEBGPU;
        }
        let instance = wgpu::Instance::new(descriptor);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|error| backend(format!("cannot request a wgpu hardware adapter: {error}")))?;
        let info = adapter.get_info();
        let hardware = matches!(
            info.device_type,
            wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
        );
        // WebGPU intentionally hides hardware identity behind Other. Its
        // fallback adapters are reported as Cpu by wgpu, never accepted here.
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        let hardware = hardware || info.device_type == wgpu::DeviceType::Other;
        if !hardware {
            return Err(backend(format!(
                "wgpu requires a hardware GPU, not {:?} ({})",
                info.device_type, info.name
            )));
        }
        let limits = wgpu::Limits::default().using_alignment(adapter.limits());
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("anr-wgpu DFN3"),
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|error| backend(format!("cannot create a wgpu device: {error}")))?;
        // Until Gpu takes ownership, cancellation or an initialization failure
        // must destroy browser resources without relying on JavaScript GC.
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        let mut initializing = InitializingDevice(Some(device.clone()));
        let errors = DeviceErrors::default();
        let uncaptured = errors.clone();
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            uncaptured.record(error.to_string());
        }));
        let lost = errors.clone();
        device.set_device_lost_callback(move |reason, message| {
            lost.record(format!("device lost ({reason:?}): {message}"));
        });
        // Pop every scope before awaiting: separate browser streams may run
        // while these promises resolve, and must not inherit this scope stack.
        let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let result = Self::allocate(device.clone(), queue, model, errors.clone());
        let validation = validation.pop();
        let internal = internal.pop();
        let memory = memory.pop();
        let validation = validation.await;
        let internal = internal.await;
        let memory = memory.await;
        if let Some(error) = [validation, internal, memory].into_iter().flatten().next() {
            return Err(backend(format!("wgpu initialization failed: {error}")));
        }
        errors.check()?;
        let core = result?;
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            initializing.0 = None;
        }
        Ok(core)
    }

    fn allocate(
        device: wgpu::Device,
        queue: wgpu::Queue,
        mut model: Model,
        errors: DeviceErrors,
    ) -> Result<Self> {
        let limits = device.limits();
        let stride = PARAMETER_BYTES
            .div_ceil(limits.min_uniform_buffer_offset_alignment)
            .checked_mul(limits.min_uniform_buffer_offset_alignment)
            .ok_or_else(|| backend("wgpu dynamic parameter stride overflow"))?;
        let steps = u32::try_from(model.dispatches.len())
            .map_err(|_| backend("wgpu dynamic parameter count overflow"))?;
        let parameter_size = steps
            .checked_mul(stride)
            .ok_or_else(|| backend("wgpu dynamic parameter offsets exceed u32"))?;
        let weight_size = model.weights.len() as u64 * 4;
        let scratch_size = u64::from(model.scratch_len) * 4;
        for size in [weight_size, STATE_BYTES, scratch_size] {
            if size > limits.max_storage_buffer_binding_size || size > limits.max_buffer_size {
                return Err(backend(format!(
                    "wgpu DFN3 storage requires {size} bytes, exceeding device limits"
                )));
            }
        }
        if u64::from(parameter_size) > limits.max_buffer_size
            || model
                .dispatches
                .iter()
                .any(|step| step.groups > limits.max_compute_workgroups_per_dimension)
        {
            return Err(backend("wgpu DFN3 schedule exceeds device limits"));
        }

        let weights = initialized_buffer(
            &device,
            &queue,
            &errors,
            "DFN3 weights",
            bytemuck::cast_slice(&model.weights),
            wgpu::BufferUsages::STORAGE,
        )?;
        // Bind groups retain the uploaded weight buffer; no CPU copy is needed.
        drop(std::mem::take(&mut model.weights));
        let parameter_bytes = model.parameter_bytes(stride);
        let parameters = initialized_buffer(
            &device,
            &queue,
            &errors,
            "DFN3 dynamic parameters",
            &parameter_bytes,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        )?;
        let states = ["DFN3 state A", "DFN3 state B"].map(|label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: STATE_BYTES,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let scratch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DFN3 scratch"),
            size: scratch_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DFN3 reusable readback"),
            size: FRAME_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        errors.check()?;

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DFN3 four storage buffers and dynamic uniform"),
            entries: &[
                storage_layout(0, true),
                storage_layout(1, true),
                storage_layout(2, false),
                storage_layout(3, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: NonZeroU64::new(u64::from(PARAMETER_BYTES)),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DFN3 pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        errors.check()?;
        let pipelines = model
            .dispatches
            .iter()
            .map(|step| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("DFN3 specialized WGSL"),
                    source: wgpu::ShaderSource::Wgsl(
                        anr_dfn3::shaders::specialized_wgsl(step).into(),
                    ),
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("DFN3 fixed dispatch pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        errors.check()?;
        let bind_groups = [0, 1].map(|current| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DFN3 recurrent state direction"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: weights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: states[current].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: states[current ^ 1].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: scratch.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &parameters,
                            offset: 0,
                            size: NonZeroU64::new(u64::from(PARAMETER_BYTES)),
                        }),
                    },
                ],
            })
        });
        errors.check()?;
        Ok(Self {
            device,
            queue,
            states,
            scratch,
            parameters,
            readback,
            pipelines,
            bind_groups,
            model,
            stride,
            current: 0,
            errors,
        })
    }

    pub(super) fn submit_frame(
        &mut self,
        input: &[f32; FRAME_SIZE],
        attenuation: f32,
        reset: bool,
    ) -> Result<wgpu::SubmissionIndex> {
        self.errors.check()?;
        let input_offset = u64::from(self.model.input) * 4;
        self.queue
            .write_buffer(&self.scratch, input_offset, bytemuck::cast_slice(input));
        if self.model.update_attenuation(attenuation) {
            let index = self.model.attenuation_dispatch;
            self.queue.write_buffer(
                &self.parameters,
                index as u64 * u64::from(self.stride) + 22 * 4,
                bytemuck::cast_slice(&self.model.dispatches[index].params[22..24]),
            );
        }
        let current = if reset { 0 } else { self.current };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("DFN3 complete frame"),
            });
        if reset {
            for state in &self.states {
                encoder.clear_buffer(state, 0, None);
            }
            // Queue writes execute before the encoder. Clear scratch on both
            // sides of this frame's fresh input rather than erasing that input.
            if input_offset != 0 {
                encoder.clear_buffer(&self.scratch, 0, Some(input_offset));
            }
            let after_input = input_offset + FRAME_BYTES;
            if after_input < self.scratch.size() {
                encoder.clear_buffer(&self.scratch, after_input, None);
            }
        }
        for (index, step) in self.model.dispatches.iter().enumerate() {
            // A pass boundary separates each producer/consumer. In particular,
            // scratch remains one read-write binding, never aliased as an SRV.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DFN3 dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines[index]);
            pass.set_bind_group(0, &self.bind_groups[current], &[index as u32 * self.stride]);
            pass.dispatch_workgroups(step.groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.scratch,
            u64::from(self.model.output) * 4,
            &self.readback,
            0,
            FRAME_BYTES,
        );
        let submission = self.queue.submit([encoder.finish()]);
        self.errors.check()?;
        Ok(submission)
    }

    pub(super) fn read_frame(&self, output: &mut [f32; FRAME_SIZE]) -> Result<()> {
        self.errors.check()?;
        let bytes = self
            .readback
            .slice(..)
            .get_mapped_range()
            .map_err(|error| backend(format!("wgpu readback range failed: {error}")))?;
        bytemuck::cast_slice_mut(output).copy_from_slice(&bytes);
        Ok(())
    }
}

fn storage_layout(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(4),
        },
        count: None,
    }
}

fn initialized_buffer(
    device: &wgpu::Device,
    _queue: &wgpu::Queue,
    errors: &DeviceErrors,
    label: &str,
    bytes: &[u8],
    usage: wgpu::BufferUsages,
) -> Result<wgpu::Buffer> {
    // mappedAtCreation can throw a JavaScript RangeError on allocation failure,
    // bypassing GPU error scopes. Browser uploads use queue writes instead.
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        _queue.write_buffer(&buffer, 0, bytes);
        errors.check()?;
        Ok(buffer)
    }
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage,
            mapped_at_creation: true,
        });
        errors.check()?;
        {
            let mut mapped = buffer
                .get_mapped_range_mut(..)
                .map_err(|error| backend(format!("wgpu initial upload mapping failed: {error}")))?;
            mapped.copy_from_slice(bytes);
        }
        buffer.unmap();
        errors.check()?;
        Ok(buffer)
    }
}
