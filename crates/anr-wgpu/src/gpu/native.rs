use anr_core::Result;
use anr_dfn3::{Executor, FRAME_SIZE, Model, PARAMETER_BYTES, STATE_SIZE, ShaderKind, backend};
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex, mpsc};

const FRAME_BYTES: u64 = (FRAME_SIZE * size_of::<f32>()) as u64;
const STATE_BYTES: u64 = (STATE_SIZE * size_of::<f32>()) as u64;

// Callbacks only record failures. The worker observes them before publishing
// output; validation, allocation, internal errors, and device loss never panic.
#[derive(Clone, Default)]
struct DeviceErrors(Arc<Mutex<Option<String>>>);

impl DeviceErrors {
    fn record(&self, message: String) {
        let mut first = self.0.lock().unwrap_or_else(|error| error.into_inner());
        if first.is_none() {
            *first = Some(message);
        }
    }

    fn check(&self) -> Result<()> {
        let first = self.0.lock().unwrap_or_else(|error| error.into_inner());
        match first.as_ref() {
            Some(message) => Err(backend(format!("wgpu: {message}"))),
            None => Ok(()),
        }
    }
}

// Constructed, used, and dropped only by the shared Denoiser's private worker.
pub(crate) struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    states: [wgpu::Buffer; 2],
    scratch: wgpu::Buffer,
    parameters: wgpu::Buffer,
    readback: wgpu::Buffer,
    pipelines: [wgpu::ComputePipeline; 3],
    bind_groups: [wgpu::BindGroup; 2],
    model: Model,
    stride: u32,
    current: usize,
    errors: DeviceErrors,
    completion: mpsc::SyncSender<std::result::Result<(), wgpu::BufferAsyncError>>,
    completed: mpsc::Receiver<std::result::Result<(), wgpu::BufferAsyncError>>,
}

impl Gpu {
    pub(crate) fn new(mut model: Model) -> Result<Self> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .map_err(|error| backend(format!("cannot request a native wgpu adapter: {error}")))?;
        let info = adapter.get_info();
        if !matches!(
            info.device_type,
            wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
        ) {
            return Err(backend(format!(
                "wgpu requires a hardware GPU, not {:?} ({})",
                info.device_type, info.name
            )));
        }
        let limits = wgpu::Limits::default().using_alignment(adapter.limits());
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("anr-wgpu DFN3"),
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(|error| backend(format!("cannot create a native wgpu device: {error}")))?;
        let errors = DeviceErrors::default();
        let uncaptured = errors.clone();
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            uncaptured.record(error.to_string());
        }));
        let lost = errors.clone();
        device.set_device_lost_callback(move |reason, message| {
            lost.record(format!("device lost ({reason:?}): {message}"));
        });

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
        let pipelines = ShaderKind::ALL.map(|kind| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("DFN3 canonical WGSL"),
                source: wgpu::ShaderSource::Wgsl(anr_dfn3::shaders::wgsl(kind).into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("DFN3 compute pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        });
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
        let (completion, completed) = mpsc::sync_channel(1);
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
            completion,
            completed,
        })
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
            pass.set_pipeline(&self.pipelines[step.shader.index()]);
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
        let completion = self.completion.clone();
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                // One outstanding map at a time; this bounded channel is reused
                // across all frames and must never block the wgpu poll thread.
                let _ = completion.try_send(result);
            });
        let waited = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        });
        let result = (|| {
            waited.map_err(|error| backend(format!("wgpu frame wait failed: {error}")))?;
            self.errors.check()?;
            self.completed
                .try_recv()
                .map_err(|error| backend(format!("wgpu map completion unavailable: {error}")))?
                .map_err(|error| backend(format!("wgpu readback map failed: {error}")))?;
            let bytes = self
                .readback
                .slice(..)
                .get_mapped_range()
                .map_err(|error| backend(format!("wgpu readback range failed: {error}")))?;
            bytemuck::cast_slice_mut(output).copy_from_slice(&bytes);
            Ok(())
        })();
        // Release the view above, then unmap even on a failed map/poll/error.
        self.readback.unmap();
        if result.is_err() {
            // An error can bypass receipt above. Do not leave a stale callback
            // result in the reusable channel if the caller requests a reset.
            let _ = self.completed.try_recv();
        }
        result?;
        self.current = current ^ 1;
        Ok(())
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // Also covers an execution failure after submit but before readback.
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
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
    errors: &DeviceErrors,
    label: &str,
    bytes: &[u8],
    usage: wgpu::BufferUsages,
) -> Result<wgpu::Buffer> {
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
