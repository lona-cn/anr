//! Native OpenVINO Runtime inference for the hash-locked DFN3 ONNX export.
//!
//! Each reducer owns one worker-local Core, stateful model and InferRequest.
//! Only audio and attenuation cross the host boundary each frame; recurrent
//! state stays inside the selected plugin. CPU is the explicit default, never
//! a fallback for GPU/NPU. CPU/GPU request FP32 and accuracy-oriented latency
//! inference. NPU hardware may execute FP16 despite FP32 model inputs; NPU is
//! not advertised as meeting the FP32 reference tolerance.
//!
//! Building requires the official OpenVINO C++ SDK via `OPENVINO_ROOT` or
//! `OpenVINO_DIR`. Official Python wheel include/libs layouts are accepted.
//! At runtime deploy `anr_openvino_native` beside the executable, plus the
//! official Runtime, ONNX frontend, and selected device plugin.
//! `ANR_OPENVINO_SHIM` and `ANR_OPENVINO_RUNTIME` override the shim and Runtime
//! shared-library paths. Build-tree paths are also searched, so local SDK
//! builds need no global installation.
mod ffi;

use anr_core::{NoiseReduction, Result};
use anr_dfn3::{Denoiser, model_bytes};
use std::{fmt, path::Path, str::FromStr, sync::mpsc};

/// An explicit OpenVINO hardware plugin; automatic and heterogeneous fallback
/// plugins are deliberately unavailable.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Device {
    /// CPU plugin (the default).
    #[default]
    Cpu,
    /// GPU plugin with the given physical device index.
    Gpu(u32),
    /// NPU plugin. Native hardware arithmetic may be FP16.
    Npu,
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => f.write_str("CPU"),
            Self::Gpu(index) => write!(f, "GPU.{index}"),
            Self::Npu => f.write_str("NPU"),
        }
    }
}

/// An invalid explicit device selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParseDeviceError;
impl fmt::Display for ParseDeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("expected CPU, GPU, GPU.<unsigned device index>, or NPU; AUTO/HETERO and fallback lists are not supported")
    }
}
impl std::error::Error for ParseDeviceError {}
impl FromStr for Device {
    type Err = ParseDeviceError;
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("CPU") {
            Ok(Self::Cpu)
        } else if value.eq_ignore_ascii_case("GPU") {
            Ok(Self::Gpu(0))
        } else if value.eq_ignore_ascii_case("NPU") {
            Ok(Self::Npu)
        } else if value.len() > 4
            && value
                .get(..4)
                .is_some_and(|s| s.eq_ignore_ascii_case("GPU."))
        {
            let index = &value[4..];
            if !index.bytes().all(|b| b.is_ascii_digit()) {
                return Err(ParseDeviceError);
            }
            index.parse().map(Self::Gpu).map_err(|_| ParseDeviceError)
        } else {
            Err(ParseDeviceError)
        }
    }
}

/// One continuous 48 kHz mono stream. Calls wait for inference completion.
/// Partial frames are zero padded; reset preserves attenuation. Validation
/// failures preserve output/state. Recreate after a native inference error.
/// Runtime, plugin, unsupported model and device errors never trigger fallback.
pub struct DFN3 {
    inner: Denoiser,
    device: Device,
    execution_device: String,
}
impl DFN3 {
    /// Required mono sample rate in hertz.
    pub const SAMPLE_RATE: u32 = Denoiser::SAMPLE_RATE;
    /// Samples per complete inference frame.
    pub const FRAME_SIZE: usize = Denoiser::FRAME_SIZE;
    /// Logical packed recurrent state values.
    pub const STATE_SIZE: usize = Denoiser::STATE_SIZE;

    /// Loads the supported model on the explicit default CPU plugin.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_file_with_device(path, Device::default())
    }

    /// Validates the model before loading Runtime, then compiles for exactly
    /// the requested device. GPU indices never select CPU on failure.
    pub fn from_file_with_device(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let bytes = model_bytes(path.as_ref())?;
        let (send, receive) = mpsc::sync_channel(1);
        let inner = Denoiser::spawn(move || {
            let executor = ffi::Native::new(bytes, device)?;
            send.send(executor.execution_device().to_owned())
                .map_err(|_| anr_dfn3::backend("OpenVINO initialization receiver stopped"))?;
            Ok(executor)
        })?;
        let execution_device = receive
            .recv()
            .map_err(|_| anr_dfn3::backend("OpenVINO did not report its execution device"))?;
        Ok(Self {
            inner,
            device,
            execution_device,
        })
    }

    /// Requested explicit device.
    pub fn device(&self) -> Device {
        self.device
    }
    /// Actual device returned by the compiled model's EXECUTION_DEVICES.
    pub fn execution_device(&self) -> &str {
        &self.execution_device
    }
}
impl NoiseReduction for DFN3 {
    fn run(&mut self, sample_rate: u32, input: &[f32], output: &mut [f32]) -> Result<()> {
        self.inner.run(sample_rate, input, output)
    }
    fn factor(&self) -> f32 {
        self.inner.factor()
    }
    fn set_factor(&mut self, value: f32) {
        self.inner.set_factor(value);
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
}
