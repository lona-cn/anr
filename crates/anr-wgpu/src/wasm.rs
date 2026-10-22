use js_sys::{Float32Array, Promise};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

type Stream = Rc<RefCell<Option<crate::web::DFN3>>>;

fn busy_error() -> JsError {
    JsError::new("DFN3 already has an in-flight run")
}

struct RunPermit {
    stream: Stream,
    inner: Option<crate::web::DFN3>,
}

impl RunPermit {
    fn acquire(stream: &Stream) -> Result<Self, JsError> {
        // None is the run permit: take ownership without retaining a RefMut.
        let inner = stream.borrow_mut().take().ok_or_else(busy_error)?;
        Ok(Self {
            stream: Rc::clone(stream),
            inner: Some(inner),
        })
    }
}

impl Drop for RunPermit {
    fn drop(&mut self) {
        *self.stream.borrow_mut() = self.inner.take();
    }
}

/// One browser WebGPU stream. Each run synchronously acquires an owned permit,
/// rejecting overlap and keeping its stream alive even if the JS wrapper is freed.
#[wasm_bindgen]
pub struct DFN3 {
    stream: Stream,
}

#[wasm_bindgen]
impl DFN3 {
    #[wasm_bindgen(js_name = fromBytes)]
    pub async fn from_bytes(bytes: Vec<u8>) -> Result<DFN3, JsError> {
        let inner = crate::web::DFN3::from_bytes(&bytes)
            .await
            .map_err(|error| JsError::new(&error.to_string()))?;
        Ok(Self {
            stream: Rc::new(RefCell::new(Some(inner))),
        })
    }

    #[wasm_bindgen(unchecked_return_type = "Promise<Float32Array>")]
    pub fn run(&self, sample_rate: u32, input: Vec<f32>) -> Promise {
        // An exported async receiver would be borrowed only when first polled,
        // leaving a free-before-poll window and throwing outside Promise rejection.
        let mut permit = match RunPermit::acquire(&self.stream) {
            Ok(permit) => permit,
            Err(error) => return Promise::reject(&JsValue::from(error)),
        };
        future_to_promise(async move {
            let mut output = vec![0.0; input.len()];
            permit
                .inner
                .as_mut()
                .expect("an active run permit owns its denoiser")
                .run(sample_rate, &input, &mut output)
                .await
                .map_err(|error| JsError::new(&error.to_string()))?;
            // Copy the Rust output once into a JS-owned transferable buffer.
            let output = Float32Array::from(output.as_slice());
            drop(permit);
            Ok(output.into())
        })
    }

    pub fn factor(&self) -> Result<f32, JsError> {
        let stream = self.stream.borrow();
        let inner = stream.as_ref().ok_or_else(busy_error)?;
        Ok(inner.factor())
    }

    #[wasm_bindgen(js_name = setFactor)]
    pub fn set_factor(&self, value: f32) -> Result<(), JsError> {
        let mut stream = self.stream.borrow_mut();
        let inner = stream.as_mut().ok_or_else(busy_error)?;
        inner.set_factor(value);
        Ok(())
    }

    pub fn reset(&self) -> Result<(), JsError> {
        let mut stream = self.stream.borrow_mut();
        let inner = stream.as_mut().ok_or_else(busy_error)?;
        inner.reset();
        Ok(())
    }
}
