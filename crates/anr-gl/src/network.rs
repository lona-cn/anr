//! The hash-locked DFN3 neural graph. Only initialization builds this schedule;
//! every frame replays fixed dispatch records without allocating or reading back.
use anr_core::Result;

use crate::gpu::Kernel;
use crate::model::{Arena, Weights};

const NONE: u32 = u32::MAX;
const LINEAR: u32 = 0;
const RELU: u32 = 1;
const SIGMOID: u32 = 2;
const TANH: u32 = 3;

pub(crate) struct Network {
    pub(crate) erb_input: u32,
    pub(crate) df_input: u32,
    pub(crate) df_history: u32,
    pub(crate) df_stem: u32,
    pub(crate) gains: u32,
    pub(crate) coefs: u32,
    kernel: Kernel,
    encoder: Vec<Dispatch>,
    decoder: Vec<Dispatch>,
}

#[derive(Clone, Copy)]
struct Tensor {
    offset: u32,
    channels: u32,
    height: u32,
    width: u32,
}

impl Tensor {
    fn len(self) -> u32 {
        self.channels * self.height * self.width
    }
}

struct Dispatch {
    params: [u32; 32],
    groups: u32,
}

struct Plan<'a> {
    weights: &'a mut Weights,
    arena: &'a mut Arena,
    ops: Vec<Dispatch>,
}

// Both exported ConvTranspose nodes have output_padding=[0,1]. For these
// fixed strides this equals stride-1, including the unstrided temporal axis.
fn conv_extent(input: u32, kernel: u32, stride: u32, pad: u32, transpose: bool) -> u32 {
    if transpose {
        (input - 1) * stride + kernel + (stride - 1) - 2 * pad
    } else {
        (input + 2 * pad - kernel) / stride + 1
    }
}

impl Plan<'_> {
    fn tensor(&mut self, channels: u32, height: u32, width: u32) -> Tensor {
        Tensor {
            offset: self.arena.alloc(channels * height * width),
            channels,
            height,
            width,
        }
    }

    fn emit(&mut self, params: [u32; 32]) {
        self.ops.push(Dispatch {
            groups: params[1].div_ceil(64),
            params,
        });
    }

    fn weight(&mut self, name: &str, count: u32) -> Result<u32> {
        let actual = self.weights.get(name)?.len();
        if actual != count as usize {
            return Err(crate::backend(format!(
                "DFN3 initializer {name}: expected {count} floats, got {actual}"
            )));
        }
        self.weights.add(name)
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        input: Tensor,
        weight: &str,
        bias: Option<&str>,
        channels: u32,
        kernel: [u32; 2],
        stride: [u32; 2],
        pad: [u32; 2],
        groups: u32,
        activation: u32,
        transpose: bool,
    ) -> Result<Tensor> {
        let output = self.tensor(
            channels,
            conv_extent(input.height, kernel[0], stride[0], pad[0], transpose),
            conv_extent(input.width, kernel[1], stride[1], pad[1], transpose),
        );
        let mut p = [0; 32];
        p[0] = u32::from(transpose);
        p[1] = output.len();
        p[2] = input.offset;
        p[3] = output.offset;
        p[4] = self.weight(
            weight,
            channels * (input.channels / groups) * kernel[0] * kernel[1],
        )?;
        p[5] = match bias {
            Some(name) => self.weight(name, channels)?,
            None => NONE,
        };
        p[6] = input.channels;
        p[7] = input.height;
        p[8] = input.width;
        p[9] = channels;
        p[10] = output.height;
        p[11] = output.width;
        p[12] = kernel[0];
        p[13] = kernel[1];
        p[14] = stride[0];
        p[15] = stride[1];
        p[16] = pad[0];
        p[17] = pad[1];
        p[18] = groups;
        p[19] = activation;
        self.emit(p);
        Ok(output)
    }

    fn point(&mut self, input: Tensor, weight: &str, bias: &str, groups: u32) -> Result<Tensor> {
        self.conv(
            input,
            weight,
            Some(bias),
            64,
            [1, 1],
            [1, 1],
            [0, 0],
            groups,
            RELU,
            false,
        )
    }

    fn depth(
        &mut self,
        input: Tensor,
        weight: &str,
        stride: u32,
        transpose: bool,
    ) -> Result<Tensor> {
        self.conv(
            input,
            weight,
            None,
            64,
            [1, 3],
            [1, stride],
            [0, 1],
            64,
            LINEAR,
            transpose,
        )
    }

    fn add(&mut self, a: Tensor, b: Tensor) -> Tensor {
        assert_eq!(a.len(), b.len());
        let output = self.tensor(a.channels, a.height, a.width);
        let mut p = [0; 32];
        p[0] = 3;
        p[1] = output.len();
        p[2] = a.offset;
        p[3] = output.offset;
        p[4] = b.offset;
        self.emit(p);
        output
    }

    // ONNX transposes NCHW -> NHWC before flattening. Group boundaries are
    // contiguous in that flattened order, not channel-contiguous NCHW.
    fn flatten(&mut self, input: Tensor) -> Tensor {
        let output = self.tensor(1, 1, input.len());
        self.reorder(input, output, false);
        output
    }

    fn unflatten(&mut self, input: Tensor, channels: u32, width: u32) -> Tensor {
        assert_eq!(input.len(), channels * width);
        let output = self.tensor(channels, 1, width);
        self.reorder(output, input, true);
        output
    }

    fn reorder(&mut self, nchw: Tensor, flat: Tensor, inverse: bool) {
        let mut p = [0; 32];
        p[0] = 4;
        p[1] = nchw.len();
        p[2] = if inverse { flat.offset } else { nchw.offset };
        p[3] = if inverse { nchw.offset } else { flat.offset };
        p[6] = nchw.channels;
        p[7] = nchw.height * nchw.width;
        p[8] = u32::from(inverse);
        self.emit(p);
    }

    fn linear(
        &mut self,
        input: Tensor,
        weight: &str,
        shape: [u32; 3],
        activation: u32,
    ) -> Result<Tensor> {
        let [groups, inputs, outputs] = shape;
        assert_eq!(input.len(), groups * inputs);
        let output = self.tensor(1, 1, groups * outputs);
        let mut p = [0; 32];
        p[0] = 2;
        p[1] = output.len();
        p[2] = input.offset;
        p[3] = output.offset;
        p[4] = self.weight(weight, groups * inputs * outputs)?;
        p[6] = groups;
        p[7] = inputs;
        p[8] = outputs;
        p[19] = activation;
        self.emit(p);
        Ok(output)
    }

    fn gru(&mut self, input: Tensor, prefix: &str, second: bool, state: u32) -> Result<Tensor> {
        assert_eq!(input.len(), 256);
        let output = self.tensor(1, 1, 256);
        let names = if second {
            [
                "Constant_7_output_0",
                "Constant_8_output_0",
                "Constant_9_output_0",
            ]
        } else {
            [
                "Constant_output_0",
                "Constant_1_output_0",
                "Constant_2_output_0",
            ]
        };
        let names = names.map(|name| format!("{prefix}/{name}"));
        let w = self.weights.get(&names[0])?;
        let r = self.weights.get(&names[1])?;
        let bias = self.weights.get(&names[2])?;
        for (name, actual, expected) in [
            (&names[0], w.len(), 3 * 256 * 256),
            (&names[1], r.len(), 3 * 256 * 256),
            (&names[2], bias.len(), 6 * 256),
        ] {
            if actual != expected {
                return Err(crate::backend(format!(
                    "DFN3 initializer {name}: expected {expected} floats, got {actual}"
                )));
            }
        }
        // Each lane computes one output j. Store [input k][six gates][j]
        // so adjacent lanes read adjacent weights in all six dot products.
        // Only this packed copy is uploaded; the raw W/R tensors are not added.
        let mut packed = Vec::with_capacity(6 * 256 * 256 + 6 * 256);
        for k in 0..256 {
            for matrix in [w, r] {
                for gate in 0..3 {
                    for j in 0..256 {
                        packed.push(matrix[(gate * 256 + j) * 256 + k]);
                    }
                }
            }
        }
        packed.extend_from_slice(bias);
        let packed_offset = self.weights.push(&packed);
        let mut p = [0; 32];
        p[0] = 5;
        p[1] = 256;
        p[2] = input.offset;
        p[3] = output.offset;
        p[4] = packed_offset;
        p[6] = 256;
        p[7] = 256;
        p[8] = state;
        p[10] = packed_offset + 6 * 256 * 256;
        self.emit(p);
        Ok(output)
    }
}

impl Network {
    pub(crate) unsafe fn new(
        gl: &glow::Context,
        weights: &mut Weights,
        arena: &mut Arena,
    ) -> Result<Self> {
        let mut plan = Plan {
            weights,
            arena,
            ops: Vec::new(),
        };
        let erb_input = plan.tensor(1, 3, 32);
        let df_input = plan.tensor(2, 3, 96);
        let df_history = plan.tensor(64, 5, 96);

        let e0 = plan.conv(
            erb_input,
            "onnx::Conv_1152",
            Some("onnx::Conv_1153"),
            64,
            [3, 3],
            [1, 1],
            [0, 1],
            1,
            RELU,
            false,
        )?;
        let e1 = plan.depth(e0, "enc.erb_conv1.0.weight", 2, false)?;
        let e1 = plan.point(e1, "onnx::Conv_1155", "onnx::Conv_1156", 1)?;
        let e2 = plan.depth(e1, "enc.erb_conv2.0.weight", 2, false)?;
        let e2 = plan.point(e2, "onnx::Conv_1158", "onnx::Conv_1159", 1)?;
        let e3 = plan.depth(e2, "enc.erb_conv3.0.weight", 1, false)?;
        let e3 = plan.point(e3, "onnx::Conv_1161", "onnx::Conv_1162", 1)?;
        let erb_embedding = plan.flatten(e3);

        let d0 = plan.conv(
            df_input,
            "enc.df_conv0.0.weight",
            None,
            64,
            [3, 3],
            [1, 1],
            [0, 1],
            2,
            LINEAR,
            false,
        )?;
        let df_stem = plan.point(d0, "onnx::Conv_1164", "onnx::Conv_1165", 1)?;
        let d1 = plan.depth(df_stem, "enc.df_conv1.0.weight", 2, false)?;
        let d1 = plan.point(d1, "onnx::Conv_1167", "onnx::Conv_1168", 1)?;
        let d1 = plan.flatten(d1);
        let df_embedding = plan.linear(d1, "enc.df_fc_emb.0.weight", [32, 96, 16], RELU)?;
        let embedding = plan.add(erb_embedding, df_embedding);
        let embedding = plan.linear(
            embedding,
            "enc.emb_gru.linear_in.0.weight",
            [16, 32, 16],
            RELU,
        )?;
        let embedding = plan.gru(embedding, "/enc/emb_gru", false, 44024)?;
        let embedding = plan.linear(
            embedding,
            "enc.emb_gru.linear_out.0.weight",
            [16, 16, 32],
            RELU,
        )?;
        let encoder = std::mem::take(&mut plan.ops);

        let erb = plan.linear(
            embedding,
            "erb_dec.emb_gru.linear_in.0.weight",
            [16, 32, 16],
            RELU,
        )?;
        let erb = plan.gru(erb, "/erb_dec/emb_gru", false, 44280)?;
        let erb = plan.gru(erb, "/erb_dec/emb_gru", true, 44536)?;
        let erb = plan.linear(
            erb,
            "erb_dec.emb_gru.linear_out.0.weight",
            [16, 16, 32],
            RELU,
        )?;
        let erb = plan.unflatten(erb, 64, 8);
        let skip3 = plan.point(e3, "onnx::Conv_1170", "onnx::Conv_1171", 64)?;
        let erb = plan.add(skip3, erb);
        let erb = plan.depth(erb, "erb_dec.convt3.0.weight", 1, false)?;
        let erb = plan.point(erb, "onnx::Conv_1173", "onnx::Conv_1174", 1)?;
        let skip2 = plan.point(e2, "onnx::Conv_1176", "onnx::Conv_1177", 64)?;
        let erb = plan.add(skip2, erb);
        let erb = plan.depth(erb, "erb_dec.convt2.0.weight", 2, true)?;
        let erb = plan.point(erb, "onnx::Conv_1179", "onnx::Conv_1180", 1)?;
        let skip1 = plan.point(e1, "onnx::Conv_1182", "onnx::Conv_1183", 64)?;
        let erb = plan.add(skip1, erb);
        let erb = plan.depth(erb, "erb_dec.convt1.0.weight", 2, true)?;
        let erb = plan.point(erb, "onnx::Conv_1185", "onnx::Conv_1186", 1)?;
        let skip0 = plan.point(e0, "onnx::Conv_1188", "onnx::Conv_1189", 64)?;
        let erb = plan.add(skip0, erb);
        let gains = plan.conv(
            erb,
            "onnx::Conv_1191",
            Some("ortshared_11_1_1"),
            1,
            [1, 3],
            [1, 1],
            [0, 1],
            1,
            SIGMOID,
            false,
        )?;

        let df = plan.linear(
            embedding,
            "df_dec.df_gru.linear_in.0.weight",
            [8, 64, 32],
            RELU,
        )?;
        let df = plan.gru(df, "/df_dec/df_gru", false, 44792)?;
        let df = plan.gru(df, "/df_dec/df_gru", true, 45048)?;
        let skip = plan.linear(embedding, "df_dec.df_skip.weight", [16, 32, 16], LINEAR)?;
        let df = plan.add(df, skip);
        let df = plan.linear(df, "df_dec.df_out.0.weight", [16, 16, 60], TANH)?;
        let temporal = plan.conv(
            df_history,
            "df_dec.df_convp.0.weight",
            None,
            10,
            [5, 1],
            [1, 1],
            [0, 0],
            2,
            LINEAR,
            false,
        )?;
        let temporal = plan.conv(
            temporal,
            "onnx::Conv_1194",
            Some("onnx::Conv_1195"),
            10,
            [1, 1],
            [1, 1],
            [0, 0],
            1,
            RELU,
            false,
        )?;
        let coefs = plan.tensor(5, 96, 2);
        let mut p = [0; 32];
        p[0] = 6;
        p[1] = coefs.len();
        p[2] = df.offset;
        p[3] = coefs.offset;
        p[4] = temporal.offset;
        plan.emit(p);
        let decoder = plan.ops;

        // Compile last so a missing/incorrect initializer cannot leak a GL program.
        let kernel = unsafe { Kernel::new(gl, include_str!("shaders/network.comp"))? };
        Ok(Self {
            erb_input: erb_input.offset,
            df_input: df_input.offset,
            df_history: df_history.offset,
            df_stem: df_stem.offset,
            gains: gains.offset,
            coefs: coefs.offset,
            kernel,
            encoder,
            decoder,
        })
    }

    pub(crate) unsafe fn encode(&self, gl: &glow::Context) {
        for op in &self.encoder {
            unsafe { self.kernel.dispatch(gl, &op.params, op.groups) };
        }
    }

    pub(crate) unsafe fn decode(&self, gl: &glow::Context) {
        for op in &self.decoder {
            unsafe { self.kernel.dispatch(gl, &op.params, op.groups) };
        }
    }

    pub(crate) unsafe fn destroy(&self, gl: &glow::Context) {
        unsafe { self.kernel.destroy(gl) };
    }
}
