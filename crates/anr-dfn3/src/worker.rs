use crate::{FRAME_SIZE, SAMPLE_RATE, STATE_SIZE, backend};
use anr_core::{Error, NoiseReduction, Result};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};

/// A worker-local native executor. It need not be Send: construction, frame
/// execution and destruction all happen on the same private thread.
pub trait Executor: 'static {
    /// Processes exactly one frame, making its output CPU-readable on success.
    fn frame(
        &mut self,
        input: &[f32; FRAME_SIZE],
        output: &mut [f32; FRAME_SIZE],
        attenuation: f32,
        reset: bool,
    ) -> Result<()>;
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

/// Synchronous mono stream with preallocated frame exchange and worker-owned
/// native resources. Used by each backend's public DFN3 type.
pub struct Denoiser {
    requests: SyncSender<Request>,
    results: Receiver<Box<Frame>>,
    worker: Option<JoinHandle<()>>,
    frame: Option<Box<Frame>>,
    attenuation: f32,
    reset_pending: bool,
}
impl Denoiser {
    /// Required sample rate.
    pub const SAMPLE_RATE: u32 = SAMPLE_RATE;
    /// Samples per complete model step.
    pub const FRAME_SIZE: usize = FRAME_SIZE;
    /// Packed state values.
    pub const STATE_SIZE: usize = STATE_SIZE;
    /// Initializes an executor on its private thread. Native handles never
    /// cross threads; only the initialization data must be Send.
    pub fn spawn<E: Executor>(
        initialize: impl FnOnce() -> Result<E> + Send + 'static,
    ) -> Result<Self> {
        let (requests, receive) = mpsc::sync_channel(1);
        let (reply, results) = mpsc::sync_channel(1);
        let (ready, initialized) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("anr-compute".into())
            .spawn(move || {
                let mut gpu = match initialize() {
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
                Err(backend("GPU 工作线程初始化失败"))
            }
        }
    }
}
impl NoiseReduction for Denoiser {
    fn run(&mut self, sample_rate: u32, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.is_empty() {
            return Err(Error::EmptyInput);
        }
        if sample_rate != SAMPLE_RATE {
            return Err(Error::InvalidSampleRate {
                expected: SAMPLE_RATE,
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
                .ok_or_else(|| backend("GPU 工作线程已停止；请重新创建降噪器"))?;
            frame.input[..input.len()].copy_from_slice(input);
            frame.input[input.len()..].fill(0.0);
            frame.attenuation = self.attenuation;
            frame.reset = self.reset_pending;
            self.requests
                .send(Request::Frame(frame))
                .map_err(|_| backend("GPU 工作线程已停止；请重新创建降噪器"))?;
            let mut frame = self
                .results
                .recv()
                .map_err(|_| backend("GPU 工作线程没有返回结果；请重新创建降噪器"))?;
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
impl Drop for Denoiser {
    fn drop(&mut self) {
        let _ = self.requests.send(Request::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
