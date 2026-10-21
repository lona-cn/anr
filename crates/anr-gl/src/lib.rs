//! Model-specific DFN3 inference using desktop OpenGL 4.3 compute shaders.
//!
//! Loads the supported external ONNX file only as a hash-locked parameter
//! container. No ONNX Runtime, CPU neural kernels, Python, or model download is
//! used. A private worker owns the current context and resident GPU state.
mod context;
mod dsp;
mod gpu;
mod model;
mod network;

use anr_core::{Error, NoiseReduction, Result};
use std::path::Path;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};

const FRAME_SIZE: usize = 480;
const STATE_SIZE: usize = 45304;

pub(crate) fn backend(message: impl Into<String>) -> Error {
    Error::backend(std::io::Error::other(message.into()))
}

struct Frame {
    input: [f32; FRAME_SIZE],
    output: [f32; FRAME_SIZE],
    attenuation: f32,
    reset: bool,
    error: Option<Error>,
}
enum Request {
    Frame(Box<Frame>),
    Shutdown,
}

/// One continuous 48 kHz mono stream, with a thread-owned OpenGL context.
///
/// `run` is synchronous: it returns only after the GPU output has been read.
/// Calls shorter than 480 samples pad and advance one complete model frame.
/// Input validation errors leave output/state unchanged; after an execution
/// error reset before reuse. No model-delay compensation or end flush is added.
pub struct DFN3 {
    requests: SyncSender<Request>,
    results: Receiver<Box<Frame>>,
    worker: Option<JoinHandle<()>>,
    frame: Option<Box<Frame>>,
    attenuation: f32,
    reset_pending: bool,
}
impl DFN3 {
    /// Required mono sample rate.
    pub const SAMPLE_RATE: u32 = 48_000;
    /// Samples consumed by one model frame.
    pub const FRAME_SIZE: usize = FRAME_SIZE;
    /// Recurrent audio/feature/network state values in this model export.
    pub const STATE_SIZE: usize = STATE_SIZE;

    /// Loads the supported model and initializes a native compute context.
    ///
    /// Requires desktop OpenGL 4.3 core on Windows or hardware EGL on Linux.
    /// Other models/platforms produce an error; no CPU fallback is performed.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let weights = model::Weights::load(path.as_ref())?;
        let (requests, receive) = mpsc::sync_channel(1);
        let (reply, results) = mpsc::sync_channel(1);
        let (ready, initialized) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("anr-gl".into())
            .spawn(move || {
                let mut gpu = match gpu::Gpu::new(weights) {
                    Ok(gpu) => gpu,
                    Err(error) => {
                        let _ = ready.send(Err(error));
                        return;
                    }
                };
                if ready.send(Ok(())).is_err() {
                    return;
                }
                while let Ok(request) = receive.recv() {
                    match request {
                        Request::Shutdown => break,
                        Request::Frame(mut frame) => {
                            frame.error = gpu
                                .frame(
                                    &frame.input,
                                    &mut frame.output,
                                    frame.attenuation,
                                    frame.reset,
                                )
                                .err();
                            if reply.send(frame).is_err() {
                                break;
                            }
                        }
                    }
                }
            })
            .map_err(Error::backend)?;
        match initialized.recv() {
            Ok(Ok(())) => Ok(Self {
                requests,
                results,
                worker: Some(worker),
                frame: Some(Box::new(Frame {
                    input: [0.0; FRAME_SIZE],
                    output: [0.0; FRAME_SIZE],
                    attenuation: 3.0,
                    reset: false,
                    error: None,
                })),
                attenuation: 3.0,
                reset_pending: false,
            }),
            Ok(Err(error)) => {
                let _ = worker.join();
                Err(error)
            }
            Err(_) => {
                let _ = worker.join();
                Err(backend("OpenGL 工作线程初始化失败"))
            }
        }
    }
}
impl NoiseReduction for DFN3 {
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
        for (input, output) in input.chunks(FRAME_SIZE).zip(output.chunks_mut(FRAME_SIZE)) {
            let mut frame = self
                .frame
                .take()
                .ok_or_else(|| backend("OpenGL 工作线程已停止；请重新创建降噪器"))?;
            frame.input[..input.len()].copy_from_slice(input);
            frame.input[input.len()..].fill(0.0);
            frame.attenuation = self.attenuation;
            frame.reset = self.reset_pending;
            self.requests
                .send(Request::Frame(frame))
                .map_err(|_| backend("OpenGL 工作线程已停止；请重新创建降噪器"))?;
            let mut frame = self
                .results
                .recv()
                .map_err(|_| backend("OpenGL 工作线程没有返回结果；请重新创建降噪器"))?;
            if let Some(error) = frame.error.take() {
                self.reset_pending = true;
                self.frame = Some(frame);
                return Err(error);
            }
            output.copy_from_slice(&frame.output[..output.len()]);
            self.reset_pending = false;
            self.frame = Some(frame);
        }
        Ok(())
    }
    fn factor(&self) -> f32 {
        (self.attenuation / 60.0 * 100.0).round()
    }
    fn set_factor(&mut self, value: f32) {
        if !value.is_nan() {
            self.attenuation = (value.clamp(0.0, 100.0) / 100.0 * 60.0).round();
        }
    }
    fn reset(&mut self) {
        self.reset_pending = true;
    }
}
impl Drop for DFN3 {
    fn drop(&mut self) {
        let _ = self.requests.send(Request::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
