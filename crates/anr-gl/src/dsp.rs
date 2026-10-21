use crate::gpu::Kernel;
use crate::model::{Arena, Weights};
use crate::network::Network;
use anr_core::Result;

pub(crate) struct Dsp {
    kernel: Kernel,
    fft: Kernel,
    params: [u32; 32],
    forward: [u32; 32],
    inverse: [u32; 32],
}
impl Dsp {
    pub(crate) unsafe fn new(
        gl: &glow::Context,
        weights: &mut Weights,
        arena: &mut Arena,
        network: &Network,
        input: u32,
        output: u32,
    ) -> Result<Self> {
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
        let kernel = unsafe { Kernel::new(gl, include_str!("shaders/dsp.comp"))? };
        let fft = match unsafe { Kernel::new(gl, include_str!("shaders/fft.comp")) } {
            Ok(fft) => fft,
            Err(error) => {
                unsafe { kernel.destroy(gl) };
                return Err(error);
            }
        };
        Ok(Self {
            kernel,
            fft,
            params: p,
            forward,
            inverse,
        })
    }

    unsafe fn dispatch(&mut self, gl: &glow::Context, mode: u32, count: u32) {
        self.params[0] = mode;
        self.params[1] = count;
        unsafe { self.kernel.dispatch(gl, &self.params, count.div_ceil(128)) };
    }

    pub(crate) unsafe fn prepare(&mut self, gl: &glow::Context) {
        unsafe {
            self.dispatch(gl, 0, 960);
            self.fft.dispatch(gl, &self.forward, 1);
            self.dispatch(gl, 1, 128);
            self.dispatch(gl, 2, 6734);
        }
    }
    pub(crate) unsafe fn history(&mut self, gl: &glow::Context) {
        unsafe { self.dispatch(gl, 3, 30720) };
    }
    pub(crate) unsafe fn finish(&mut self, gl: &glow::Context, attenuation: f32) {
        self.params[22] = 10.0_f32.powf(-attenuation.abs() / 20.0).to_bits();
        self.params[23] = u32::from(attenuation.abs() > 0.0);
        unsafe {
            self.dispatch(gl, 4, 481);
            self.fft.dispatch(gl, &self.inverse, 1);
            self.dispatch(gl, 5, 480);
        }
    }
    pub(crate) unsafe fn destroy(&self, gl: &glow::Context) {
        unsafe {
            self.kernel.destroy(gl);
            self.fft.destroy(gl);
        }
    }
}
