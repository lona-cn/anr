use super::shared::Core;
use anr_core::Result;
use anr_dfn3::{FRAME_SIZE, Model, backend};
use std::cell::RefCell;
use std::future::poll_fn;
use std::rc::Rc;
use std::task::{Poll, Waker};

#[derive(Default)]
struct Completion {
    generation: u64,
    result: Option<std::result::Result<(), wgpu::BufferAsyncError>>,
    waker: Option<Waker>,
}

// Each map has an unmapping guard, including while its future is cancelled.
// Generations prevent a late cancelled callback from completing a later frame.
struct Mapping {
    buffer: wgpu::Buffer,
    completion: Rc<RefCell<Completion>>,
}

impl Mapping {
    fn start(buffer: &wgpu::Buffer, completion: &Rc<RefCell<Completion>>) -> Self {
        let generation = {
            let mut state = completion.borrow_mut();
            state.generation = state.generation.wrapping_add(1);
            state.result = None;
            state.waker = None;
            state.generation
        };
        let notify = Rc::clone(completion);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let waker = {
                    let mut state = notify.borrow_mut();
                    if state.generation != generation {
                        return;
                    }
                    state.result = Some(result);
                    state.waker.take()
                };
                if let Some(waker) = waker {
                    waker.wake();
                }
            });
        Self {
            buffer: buffer.clone(),
            completion: Rc::clone(completion),
        }
    }

    async fn wait(&self) -> Result<()> {
        poll_fn(|cx| {
            let mut state = self.completion.borrow_mut();
            if let Some(result) = state.result.take() {
                return Poll::Ready(
                    result.map_err(|error| backend(format!("wgpu readback map failed: {error}"))),
                );
            }
            if !state
                .waker
                .as_ref()
                .is_some_and(|waker| waker.will_wake(cx.waker()))
            {
                state.waker = Some(cx.waker().clone());
            }
            Poll::Pending
        })
        .await
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        {
            let mut state = self.completion.borrow_mut();
            state.generation = state.generation.wrapping_add(1);
            state.result = None;
            state.waker = None;
        }
        self.buffer.unmap();
    }
}

pub(crate) struct Gpu {
    core: Core,
    completion: Rc<RefCell<Completion>>,
}

impl Gpu {
    pub(crate) async fn new(model: Model) -> Result<Self> {
        Ok(Self {
            core: Core::new(model).await?,
            completion: Rc::default(),
        })
    }

    pub(crate) async fn frame(
        &mut self,
        input: &[f32; FRAME_SIZE],
        output: &mut [f32; FRAME_SIZE],
        attenuation: f32,
        reset: bool,
    ) -> Result<()> {
        self.core.errors.check()?;
        let memory = self
            .core
            .device
            .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = self
            .core
            .device
            .push_error_scope(wgpu::ErrorFilter::Internal);
        let validation = self
            .core
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let submitted = self.core.submit_frame(input, attenuation, reset);
        let mapping = submitted.map(|_| Mapping::start(&self.core.readback, &self.completion));
        // Pop before yielding, so interleaved instances never share scope stacks.
        let validation = validation.pop();
        let internal = internal.pop();
        let memory = memory.pop();
        let mapped = match mapping.as_ref() {
            Ok(mapping) => mapping.wait().await,
            Err(_) => Ok(()),
        };
        // Error-scope promises are real completion boundaries, including shader
        // validation. Merely inspecting uncaptured callbacks can race WebGPU.
        let validation = validation.await;
        let internal = internal.await;
        let memory = memory.await;
        if let Some(error) = [validation, internal, memory].into_iter().flatten().next() {
            return Err(backend(format!("wgpu frame failed: {error}")));
        }
        let mapping = mapping?;
        mapped?;
        self.core.read_frame(output)?;
        drop(mapping);
        self.core.current = if reset { 1 } else { self.core.current ^ 1 };
        Ok(())
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // The device belongs only to this stream. Destroy releases its GPU
        // resources immediately without waiting for submitted work or JS GC.
        self.core.device.destroy();
    }
}
