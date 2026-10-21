use anr_core::{Error, NoiseReduction, Result};
use ort::{
    AsPointer,
    environment::Environment,
    memory::Allocator,
    ortsys,
    session::{HasSelectedOutputs, OutputSelector, RunOptions, Session, SessionInputValue},
    value::{Outlet, Tensor, TensorElementType, ValueType},
};
use std::path::Path;

const MAX_ATTENUATION_DB: f32 = 60.0;
const INPUTS: [(&str, usize); 3] = [
    ("input_frame", DFN3::FRAME_SIZE),
    ("states", DFN3::STATE_SIZE),
    ("atten_lim_db", 1),
];
const OUTPUTS: [(&str, usize); 3] = [
    ("enhanced_audio_frame", DFN3::FRAME_SIZE),
    ("new_states", DFN3::STATE_SIZE),
    ("lsnr", 1),
];

/// Stateful DeepFilterNet3 inference for the single-frame ONNX export.
///
/// Accepts 48 kHz mono `f32` samples. Each inference consumes 480 samples
/// (10 ms); a partial final frame is zero-padded and its padding is discarded
/// from the output, but still advances the recurrent state by a full frame.
/// For continuous audio, use multiples of 480 except at the end of the stream.
/// Call [`NoiseReduction::reset`] before processing an unrelated stream.
///
/// This wrapper does not resample, decode files, compensate model latency, or
/// load the upstream multi-file DeepFilterNet ONNX export. See the crate README
/// for the required three-input/three-output model contract.
pub struct DFN3 {
    session: Session,
    input_frame: Tensor<f32>,
    // Release the non-owning output wrappers before their state allocations.
    options: [RunOptions<HasSelectedOutputs>; 2],
    states: [Tensor<f32>; 2],
    atten_lim_db: Tensor<f32>,
    input_order: [usize; 3],
    current: usize,
}

impl DFN3 {
    /// Required input sample rate in hertz.
    pub const SAMPLE_RATE: u32 = 48_000;
    /// Number of audio samples consumed per inference.
    pub const FRAME_SIZE: usize = 480;
    /// Number of recurrent state values in this ONNX export.
    pub const STATE_SIZE: usize = 45_304;

    /// Wraps a configured ORT session and initializes a new stream.
    ///
    /// Configure execution providers on the session before passing it here.
    /// The wrapper preserves the session's operator placement: unsupported or
    /// CPU-preferred operators may use CPU kernels alongside a GPU provider.
    /// Tensors are CPU-backed, including when a provider executes on a GPU;
    /// ORT handles the required transfers. Audio/state buffers are allocated
    /// once and the two state buffers alternate without a per-frame state copy.
    ///
    /// # Errors
    /// Returns [`Error::InvalidModel`] for incompatible I/O names, types, or
    /// shapes, or [`Error::Backend`] if runtime initialization fails.
    pub fn new(session: Session) -> Result<Self> {
        validate_ports(session.inputs(), &INPUTS)?;
        validate_ports(session.outputs(), &OUTPUTS)?;
        let input_order = std::array::from_fn(|i| {
            INPUTS
                .iter()
                .position(|(name, _)| *name == session.inputs()[i].name())
                .expect("validated input name")
        });
        // Never use the session's allocator: a custom GPU session must still
        // receive CPU-backed inputs and preallocated outputs.
        let allocator = Allocator::default();
        let input_frame = zeros(&allocator, Self::FRAME_SIZE)?;
        let mut states = [
            zeros(&allocator, Self::STATE_SIZE)?,
            zeros(&allocator, Self::STATE_SIZE)?,
        ];
        let attenuation_port = session
            .inputs()
            .iter()
            .find(|port| port.name() == "atten_lim_db")
            .expect("validated attenuation input");
        let attenuation_shape = match attenuation_port.dtype() {
            ValueType::Tensor { shape, .. } if shape.is_empty() => Vec::<usize>::new(),
            _ => vec![1],
        };
        let atten_lim_db =
            Tensor::from_array((attenuation_shape, vec![3.0_f32])).map_err(Error::backend)?;
        // SAFETY: these fixed-size allocations never resize or escape DFN3.
        // Options are declared after states locally and before them as fields:
        // partial construction errors and normal drop both release aliases
        // before the buffers. No fallible work follows successful construction.
        let options = unsafe {
            [
                output_options(&allocator, &mut states[1])?,
                output_options(&allocator, &mut states[0])?,
            ]
        };
        Ok(Self {
            session,
            input_frame,
            states,
            atten_lim_db,
            options,
            input_order,
            current: 0,
        })
    }

    /// Loads a model using ORT's built-in CPU execution provider.
    ///
    /// Uses one intra-op thread and disables thread-pool spinning, environment
    /// execution providers, and automatic accelerator selection. Use
    /// [`Self::new`] to supply a session with different runtime settings.
    /// The caller owns the environment; configure it with [`ort::init`] before
    /// passing it here. Dynamic builds must prepare the runtime with
    /// [`crate::initialize_runtime`] before creating that environment.
    ///
    /// # Errors
    /// Returns a backend error if the file cannot be loaded, or an invalid-model
    /// error if its I/O contract does not match this DFN3 export.
    pub fn from_file(environment: &Environment, path: impl AsRef<Path>) -> Result<Self> {
        let session = (|| -> ort::Result<Session> {
            cpu_session(environment)?
                .with_no_environment_execution_providers()?
                .with_independent_thread_pool()?
                .with_intra_threads(1)?
                .with_intra_op_spinning(false)?
                .with_inter_op_spinning(false)?
                .commit_from_file(path)
        })()
        .map_err(Error::backend)?;
        Self::new(session)
    }

    /// Sets the maximum attenuation in dB, clamped to `0..=60`.
    ///
    /// Larger values allow stronger suppression; they are not an output gain
    /// or a promise of a particular measured noise reduction. NaN leaves the
    /// previous setting unchanged; infinities saturate at the nearest bound.
    pub fn set_atten_lim_db(&mut self, value: f32) {
        if !value.is_nan() {
            self.atten_lim_db.extract_tensor_mut().1[0] = value.clamp(0.0, MAX_ATTENUATION_DB);
        }
    }

    /// Returns the configured maximum attenuation in dB.
    pub fn get_atten_lim_db(&self) -> f32 {
        self.atten_lim_db.extract_tensor().1[0]
    }
}

impl NoiseReduction for DFN3 {
    /// Processes audio, preserving model state across calls.
    ///
    /// Empty input, a non-48 kHz rate, or unequal slice lengths fail before
    /// modifying output or stream state. If inference fails after processing
    /// begins, output and state may be partially advanced; reset before reuse.
    fn run(&mut self, sample_rate: u32, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.is_empty() {
            return Err(Error::EmptyInput);
        }
        if sample_rate != Self::SAMPLE_RATE {
            return Err(Error::InvalidSampleRate {
                expected: Self::SAMPLE_RATE,
                actual: sample_rate,
            });
        }
        if input.len() != output.len() {
            return Err(Error::OutputLengthMismatch {
                input: input.len(),
                output: output.len(),
            });
        }
        for (input, output) in input
            .chunks(Self::FRAME_SIZE)
            .zip(output.chunks_mut(Self::FRAME_SIZE))
        {
            let frame = self.input_frame.extract_tensor_mut().1;
            frame[..input.len()].copy_from_slice(input);
            frame[input.len()..].fill(0.0);
            let tensors = [
                &self.input_frame,
                &self.states[self.current],
                &self.atten_lim_db,
            ];
            let inputs: [SessionInputValue<'_>; 3] = self.input_order.map(|i| tensors[i].into());
            let outputs = self
                .session
                .run_with_options(inputs, &self.options[self.current])
                .map_err(Error::backend)?;
            let audio = outputs["enhanced_audio_frame"]
                .try_extract_tensor::<f32>()
                .map_err(Error::backend)?
                .1;
            output.copy_from_slice(&audio[..output.len()]);
            self.current ^= 1;
        }
        Ok(())
    }

    /// Returns strength in `0..=100`, rounded from the attenuation setting.
    fn factor(&self) -> f32 {
        (self.get_atten_lim_db() / MAX_ATTENUATION_DB * 100.0).round()
    }

    /// Maps `0..=100` strength to `0..=60` dB, rounded to a whole dB.
    /// NaN leaves the setting unchanged; out-of-range values saturate.
    fn set_factor(&mut self, value: f32) {
        self.set_atten_lim_db((value.clamp(0.0, 100.0) / 100.0 * MAX_ATTENUATION_DB).round());
    }

    fn reset(&mut self) {
        for state in &mut self.states {
            state.extract_tensor_mut().1.fill(0.0);
        }
        self.current = 0;
    }
}

fn cpu_session(environment: &Environment) -> ort::Result<ort::session::builder::SessionBuilder> {
    let mut builder = Session::builder(environment)?;
    // CPU::build() only configures its arena, leaving automatic NPU selection
    // enabled. Select the exact built-in CPU EP instead of a preferred device.
    // Minimal ORT builds skip automatic EP selection entirely and use their
    // built-in CPU without requiring the unsupported GetEpDevices API.
    ort::ortsys![unsafe SessionOptionsSetEpSelectionPolicyDelegate(
        builder.ptr_mut(),
        Some(select_cpu),
        std::ptr::null_mut()
    )?];
    Ok(builder)
}

unsafe extern "system" fn select_cpu(
    devices: *const *const ort::sys::OrtEpDevice,
    device_count: usize,
    _model_metadata: *const ort::sys::OrtKeyValuePairs,
    _runtime_metadata: *const ort::sys::OrtKeyValuePairs,
    selected: *mut *const ort::sys::OrtEpDevice,
    capacity: usize,
    selected_count: *mut usize,
    _state: *mut std::ffi::c_void,
) -> ort::sys::OrtStatusPtr {
    // SAFETY: ORT supplies valid device/name pointers and a writable count for
    // this call. Write at most one entry within the supplied output capacity.
    // The callback retains no pointers or Rust state and lives for the process.
    unsafe {
        selected_count.write(0);
        if capacity > 0 {
            for index in 0..device_count {
                let device = *devices.add(index);
                let name = (ort::api().EpDevice_EpName)(device);
                if std::ffi::CStr::from_ptr(name).to_bytes() == b"CPUExecutionProvider" {
                    selected.write(device);
                    selected_count.write(1);
                    break;
                }
            }
        }
    }
    // A zero count is an explicit selection failure in full ORT, never a
    // request to fall back to another device. ORT reports that failure itself.
    ort::sys::OrtStatusPtr::default()
}

fn zeros(allocator: &Allocator, size: usize) -> Result<Tensor<f32>> {
    let mut tensor = Tensor::<f32>::new(allocator, [size]).map_err(Error::backend)?;
    tensor.extract_tensor_mut().1.fill(0.0);
    Ok(tensor)
}

/// # Safety
/// `state` must be a native CPU allocation from `allocator`, containing exactly
/// STATE_SIZE f32 values. It must outlive the returned wrapper without resizing.
unsafe fn state_output_alias(
    allocator: &Allocator,
    state: &mut Tensor<f32>,
) -> ort::Result<Tensor<f32>> {
    let shape = [DFN3::STATE_SIZE as i64];
    let mut value = std::ptr::null_mut();
    // The MemoryInfo describes this exact CPU allocation. ORT copies that
    // description; the data pointer remains valid because DFN3 owns the buffer.
    let memory_info = state.memory_info().ptr();
    let data = state.data_ptr_mut();
    // SAFETY: the fixed shape and byte length cover the initialized f32 buffer
    // exactly. This creates only an OrtValue wrapper; ORT does not take
    // ownership of the supplied allocation, and it does not retain shape.
    ortsys![unsafe CreateTensorWithDataAsOrtValue(
        memory_info,
        data,
        DFN3::STATE_SIZE * std::mem::size_of::<f32>(),
        shape.as_ptr(),
        shape.len(),
        ort::sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &mut value
    )?; nonNull(value)];
    // SAFETY: from_ptr takes sole ownership of the newly created OrtValue
    // wrapper, not its external buffer. This is the allocator of that buffer;
    // DFN3 separately retains its true owner until the wrapper has been dropped.
    // No aliases escape, and &mut self serializes inference and reset. The two
    // ping-pong allocations keep input state distinct from output state.
    Ok(unsafe { Tensor::from_ptr(value, Some(allocator)) })
}

/// # Safety
/// The state allocation must satisfy state_output_alias's contract and outlive
/// the returned RunOptions, including every temporary output from their runs.
unsafe fn output_options(
    allocator: &Allocator,
    next_state: &mut Tensor<f32>,
) -> Result<RunOptions<HasSelectedOutputs>> {
    // SAFETY: forwarded from this function's caller.
    let next_state =
        unsafe { state_output_alias(allocator, next_state) }.map_err(Error::backend)?;
    // Audio outputs are native CPU tensors owned directly by OutputSelector;
    // unlike the state wrapper, they need no separately retained buffer owner.
    let outputs = OutputSelector::no_default()
        .with("enhanced_audio_frame")
        .with("new_states")
        .preallocate("enhanced_audio_frame", zeros(allocator, DFN3::FRAME_SIZE)?)
        .preallocate("new_states", next_state);
    // lsnr is not consumed by NoiseReduction; do not allocate or collect it.
    Ok(RunOptions::new()
        .map_err(Error::backend)?
        .with_outputs(outputs))
}

fn validate_ports(ports: &[Outlet], expected: &[(&str, usize); 3]) -> Result<()> {
    if ports.len() != expected.len() {
        return Err(Error::InvalidModel(format!(
            "expected {} ports, found {}",
            expected.len(),
            ports.len()
        )));
    }
    for &(name, size) in expected {
        let port = ports
            .iter()
            .find(|port| port.name() == name)
            .ok_or_else(|| Error::InvalidModel(format!("missing tensor {name:?}")))?;
        if !matches!(port.dtype(), ValueType::Tensor { ty: TensorElementType::Float32, shape, .. }
            if (shape.len() == 1 && (shape[0] == size as i64 || shape[0] == -1))
                || (size == 1 && shape.is_empty()))
        {
            return Err(Error::InvalidModel(format!(
                "tensor {name:?} must be f32[{size}]{} (symbolic dimensions allowed), found {}",
                if size == 1 { " or a scalar" } else { "" },
                port.dtype()
            )));
        }
    }
    Ok(())
}
