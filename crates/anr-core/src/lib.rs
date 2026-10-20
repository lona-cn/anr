//! Backend-independent contracts for stateful audio noise reduction.
//!
//! This crate defines the processing interface and shared errors. It does not
//! load models, decode audio files, resample audio, or select a runtime.

use std::fmt;

/// A result returned by a noise-reduction operation.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// An error encountered while preparing or running a noise reducer.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The input slice contains no audio samples.
    EmptyInput,
    /// The supplied sample rate does not match the backend's required rate.
    InvalidSampleRate {
        /// The sample rate required by the backend, in hertz.
        expected: u32,
        /// The sample rate supplied by the caller, in hertz.
        actual: u32,
    },
    /// The output slice has a different length from the input slice.
    OutputLengthMismatch {
        /// The number of input samples.
        input: usize,
        /// The number of output samples.
        output: usize,
    },
    /// The model does not satisfy the backend's required interface.
    InvalidModel(String),
    /// An error from the underlying runtime or another backend dependency.
    Backend(Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    /// Wraps a backend error without losing its concrete type or error chain.
    ///
    /// The original error is available through [`std::error::Error::source`]
    /// and can be downcast to its concrete type.
    pub fn backend(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Backend(Box::new(error))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => formatter.write_str("audio input must not be empty"),
            Self::InvalidSampleRate { expected, actual } => write!(
                formatter,
                "invalid sample rate: expected {expected} Hz, received {actual} Hz"
            ),
            Self::OutputLengthMismatch { input, output } => write!(
                formatter,
                "output length mismatch: input has {input} samples, output has {output} samples"
            ),
            Self::InvalidModel(reason) => write!(formatter, "invalid model: {reason}"),
            Self::Backend(error) => write!(formatter, "backend error: {error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Backend(error) => Some(error.as_ref()),
            _ => None,
        }
    }
}

/// A stateful noise reducer for a single stream of audio samples.
///
/// Successive calls to [`run`](Self::run) belong to the same stream and may
/// depend on state accumulated from earlier calls. Call [`reset`](Self::reset)
/// before processing an unrelated stream.
///
/// Samples are supplied as `f32` values in a single-channel slice. The caller
/// provides an equally sized output slice; implementations do not resample,
/// decode files, or deinterleave channels through this interface. Required
/// sample rates, frame sizes, model latency, and handling of a partial final
/// frame are defined by each backend. In particular, a partial frame may advance
/// internal state beyond the number of samples returned to the caller.
///
/// Strength bounds and the meaning of the strength value are implementation
/// defined. This trait is object-safe and reducers can be moved between threads.
pub trait NoiseReduction: Send {
    /// Processes samples from the current stream into a caller-owned buffer.
    ///
    /// `sample_rate` is the input sample rate in hertz. `input` must be nonempty
    /// and `output` must have the same length. On success, every output sample
    /// has been written. Refer to the backend's documentation for supported
    /// sample rates, framing requirements, and latency.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmptyInput`], [`Error::InvalidSampleRate`], or
    /// [`Error::OutputLengthMismatch`] when those input constraints are not met.
    /// Validation errors must leave the output buffer and stream state
    /// unchanged. The backend defines the state and output guarantees for
    /// errors encountered after processing starts.
    fn run(&mut self, sample_rate: u32, input: &[f32], output: &mut [f32]) -> Result<()>;

    /// Returns the current noise-reduction strength in backend-defined units.
    fn factor(&self) -> f32;

    /// Sets the noise-reduction strength.
    ///
    /// Accepted bounds and handling of out-of-range or non-finite values are
    /// defined by the implementation.
    fn set_factor(&mut self, value: f32);

    /// Clears accumulated audio state to begin a new stream.
    ///
    /// The current noise-reduction strength is preserved.
    fn reset(&mut self);
}

#[cfg(test)]
mod tests;
