use crate::network::Network;
use crate::weights::{Arena, Weights};
use crate::{Dispatch, FRAME_SIZE, PARAMETER_BYTES, ShaderKind};
use anr_core::Result;
use std::path::Path;

/// Parameters and fixed dispatch schedule for the supported DFN3 export.
/// Offsets and scratch_len use f32 elements, not bytes.
pub struct Model {
    /// One-time GPU upload data. Backends can release it after upload.
    pub weights: Vec<f32>,
    /// Scratch allocation length in f32 elements.
    pub scratch_len: u32,
    /// Input frame offset in scratch, in f32 elements.
    pub input: u32,
    /// Output frame offset in scratch, in f32 elements.
    pub output: u32,
    /// Full ordered frame schedule, including FFT and state updates.
    pub dispatches: Vec<Dispatch>,
    /// The spectral-filter step whose attenuation words can change.
    pub attenuation_dispatch: usize,
    attenuation: f32,
}
impl Model {
    /// Validates the model identity and constructs the immutable GPU layout.
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_aligned(path, 1)
    }

    /// Aligns each scratch tensor to a power of two, up to 64 f32 elements
    /// (256 bytes). The model's packed recurrent-state offsets stay unchanged.
    pub fn load_aligned(path: &Path, alignment: u32) -> Result<Self> {
        let mut weights = Weights::load(path)?;
        let mut arena = Arena::new(alignment)?;
        let input = arena.alloc(FRAME_SIZE as u32);
        let output = arena.alloc(FRAME_SIZE as u32);
        let network = Network::build(&mut weights, &mut arena)?;
        let mut p = [0; 32];
        p[2] = input;
        p[3] = output;
        p[4] = weights.add("window")?;
        p[5] = arena.alloc(960);
        p[6] = arena.alloc(962);
        p[7] = arena.alloc(32);
        p[8] = arena.alloc(192);
        p[9] = network.erb_input;
        p[10] = network.df_input;
        p[11] = network.df_history;
        p[12] = network.df_stem;
        p[13] = network.gains;
        p[14] = network.coefs;
        p[15] = arena.alloc(962);
        p[16] = arena.alloc(960);
        p[17] = weights.add("/Constant_21_output_0")?;
        p[18] = weights.add("/Constant_22_output_0")?;
        let matrix = weights.get("forward_erb_matrix")?;
        let mut bounds = [0.0; 64];
        let mut scales = [0.0; 481];
        let mut bands = [0.0; 481];
        for band in 0..32 {
            let mut start = 481;
            let mut end = 0;
            for f in 0..481 {
                let value = matrix[f * 32 + band];
                if value != 0.0 {
                    start = start.min(f);
                    end = f + 1;
                    scales[f] = value;
                    bands[f] = band as f32;
                }
            }
            bounds[band] = start as f32;
            bounds[32 + band] = end as f32;
        }
        p[19] = weights.push(&bounds);
        p[20] = weights.push(&scales);
        p[21] = weights.push(&bands);
        let twiddles: Vec<f32> = (0..960)
            .flat_map(|i| {
                let phase = -(i as f64) * std::f64::consts::TAU / 960.0;
                [phase.cos() as f32, phase.sin() as f32]
            })
            .collect();
        let twiddle_offset = weights.push(&twiddles);
        let mut forward = [0; 32];
        forward[2] = p[5];
        forward[3] = p[6];
        forward[4] = twiddle_offset;
        let mut inverse = [0; 32];
        inverse[0] = 1;
        inverse[2] = p[15];
        inverse[3] = p[16];
        inverse[4] = twiddle_offset;
        let dsp = |mode: u32, count: u32| {
            let mut params = p;
            params[0] = mode;
            params[1] = count;
            Dispatch {
                shader: ShaderKind::Dsp,
                params,
                groups: count.div_ceil(128),
            }
        };
        let mut dispatches = Vec::with_capacity(network.encoder.len() + network.decoder.len() + 8);
        dispatches.push(dsp(0, 960));
        dispatches.push(Dispatch {
            shader: ShaderKind::Fft,
            params: forward,
            groups: 1,
        });
        dispatches.push(dsp(1, 128));
        dispatches.push(dsp(2, 6734));
        dispatches.extend(network.encoder);
        dispatches.push(dsp(3, 30720));
        dispatches.extend(network.decoder);
        let attenuation_dispatch = dispatches.len();
        dispatches.push(dsp(4, 481));
        dispatches.push(Dispatch {
            shader: ShaderKind::Fft,
            params: inverse,
            groups: 1,
        });
        dispatches.push(dsp(5, 480));
        let mut model = Self {
            weights: weights.data,
            scratch_len: arena.len,
            input,
            output,
            dispatches,
            attenuation_dispatch,
            attenuation: f32::NAN,
        };
        model.update_attenuation(3.0);
        Ok(model)
    }

    /// Packs fixed parameter blocks using the backend's required slice stride.
    /// The caller supplies a valid aligned stride of at least 128 bytes.
    pub fn parameter_bytes(&self, stride: u32) -> Vec<u8> {
        assert!(stride >= PARAMETER_BYTES && stride.is_multiple_of(16));
        let mut bytes = vec![0; stride as usize * self.dispatches.len()];
        for (dispatch, slot) in self
            .dispatches
            .iter()
            .zip(bytes.chunks_exact_mut(stride as usize))
        {
            slot[..PARAMETER_BYTES as usize]
                .copy_from_slice(bytemuck::cast_slice(&dispatch.params));
        }
        bytes
    }

    /// Changes only the two attenuation words, returning whether an upload is
    /// needed. State reset never resets the user's strength setting.
    pub fn update_attenuation(&mut self, db: f32) -> bool {
        if self.attenuation == db {
            return false;
        }
        let params = &mut self.dispatches[self.attenuation_dispatch].params;
        params[22] = 10.0_f32.powf(-db.abs() / 20.0).to_bits();
        params[23] = u32::from(db.abs() > 0.0);
        self.attenuation = db;
        true
    }
}
