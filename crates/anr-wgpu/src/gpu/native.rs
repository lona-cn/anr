use super::shared::Core;
use anr_core::Result;
use anr_dfn3::{Executor, FRAME_SIZE, Model, backend};
use std::sync::mpsc;

// Only the shared Denoiser's private worker constructs and drops this executor.
pub(crate) struct Gpu {
    core: Core,
    completion: mpsc::SyncSender<std::result::Result<(), wgpu::BufferAsyncError>>,
    completed: mpsc::Receiver<std::result::Result<(), wgpu::BufferAsyncError>>,
}

impl Gpu {
    pub(crate) fn new(model: Model) -> Result<Self> {
        let core = pollster::block_on(Core::new(model))?;
        let (completion, completed) = mpsc::sync_channel(1);
        Ok(Self {
            core,
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
        let submission = self.core.submit_frame(input, attenuation, reset)?;
        let completion = self.completion.clone();
        self.core
            .readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                // Never block the device's poll thread.
                let _ = completion.try_send(result);
            });
        let waited = self.core.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        });
        let result = (|| {
            waited.map_err(|error| backend(format!("wgpu frame wait failed: {error}")))?;
            self.core.errors.check()?;
            self.completed
                .try_recv()
                .map_err(|error| backend(format!("wgpu map completion unavailable: {error}")))?
                .map_err(|error| backend(format!("wgpu readback map failed: {error}")))?;
            self.core.read_frame(output)
        })();
        self.core.readback.unmap();
        if result.is_err() {
            let _ = self.completed.try_recv();
        }
        result?;
        self.core.current = if reset { 1 } else { self.core.current ^ 1 };
        Ok(())
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        let _ = self.core.device.poll(wgpu::PollType::wait_indefinitely());
    }
}
