//! Denoise a 48 kHz mono stream of little-endian f32 samples (not a WAV file).
use anr::{DFN3, NoiseReduction};
use std::{
    env,
    error::Error,
    fs,
    io::{self, BufWriter, Write},
};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args_os().collect();
    if args.len() != 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: denoise <model.onnx> <input.f32le> <output.f32le> (48 kHz, mono, raw f32; not WAV)",
        ).into());
    }
    let bytes = fs::read(&args[2])?;
    let (samples, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input length must be a multiple of 4 bytes",
        )
        .into());
    }
    let input: Vec<f32> = samples.iter().copied().map(f32::from_le_bytes).collect();
    let mut model = DFN3::from_file(&args[1])?;
    model.set_factor(80.0);
    let mut output = vec![0.0; input.len()];
    let reducer: &mut dyn NoiseReduction = &mut model;
    reducer.run(DFN3::SAMPLE_RATE, &input, &mut output)?;
    let file = fs::File::options()
        .write(true)
        .create_new(true)
        .open(&args[3])?;
    let mut writer = BufWriter::new(file);
    for sample in output {
        writer.write_all(&sample.to_le_bytes())?;
    }
    writer.flush()?;
    eprintln!(
        "Processed {} samples at 48000 Hz (mono); output: {}",
        input.len(),
        args[3].to_string_lossy()
    );
    Ok(())
}
