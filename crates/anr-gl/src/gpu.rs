use crate::context::GlContext;
use crate::dsp::Dsp;
use crate::model::{Arena, Weights};
use crate::network::Network;
use crate::{FRAME_SIZE, STATE_SIZE, backend};
use anr_core::Result;
use glow::HasContext;

pub(crate) struct Kernel {
    program: glow::NativeProgram,
    params: glow::NativeUniformLocation,
}
impl Kernel {
    pub(crate) unsafe fn new(gl: &glow::Context, source: &str) -> Result<Self> {
        unsafe {
            let shader = gl.create_shader(glow::COMPUTE_SHADER).map_err(backend)?;
            gl.shader_source(shader, source);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                let log = gl.get_shader_info_log(shader);
                gl.delete_shader(shader);
                return Err(backend(format!("无法编译 Compute Shader：{log}")));
            }
            let program = match gl.create_program() {
                Ok(program) => program,
                Err(error) => {
                    gl.delete_shader(shader);
                    return Err(backend(error));
                }
            };
            gl.attach_shader(program, shader);
            gl.link_program(program);
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
            if !gl.get_program_link_status(program) {
                let log = gl.get_program_info_log(program);
                gl.delete_program(program);
                return Err(backend(format!("无法链接 Compute Shader：{log}")));
            }
            let Some(params) = gl.get_uniform_location(program, "p[0]") else {
                gl.delete_program(program);
                return Err(backend("Compute Shader 缺少参数"));
            };
            Ok(Self { program, params })
        }
    }
    pub(crate) unsafe fn dispatch(&self, gl: &glow::Context, params: &[u32], groups: u32) {
        unsafe {
            gl.use_program(Some(self.program));
            gl.uniform_1_u32_slice(Some(&self.params), params);
            gl.dispatch_compute(groups, 1, 1);
            gl.memory_barrier(glow::SHADER_STORAGE_BARRIER_BIT);
        }
    }
    pub(crate) unsafe fn destroy(&self, gl: &glow::Context) {
        unsafe { gl.delete_program(self.program) };
    }
}

// All fields and native objects live exclusively on the worker thread.
// Drop explicitly destroys GPU objects while the native context is current.
pub(crate) struct Gpu {
    context: GlContext,
    network: Network,
    dsp: Dsp,
    buffers: [glow::NativeBuffer; 4],
    current: usize,
    input: u32,
    output: u32,
    zeros: Vec<f32>,
}
impl Gpu {
    pub(crate) fn new(mut weights: Weights) -> Result<Self> {
        let context = GlContext::new()?;
        let gl = &context.gl;
        let mut arena = Arena::default();
        let input = arena.alloc(FRAME_SIZE as u32);
        let output = arena.alloc(FRAME_SIZE as u32);
        let network = unsafe { Network::new(gl, &mut weights, &mut arena)? };
        let dsp = match unsafe { Dsp::new(gl, &mut weights, &mut arena, &network, input, output) } {
            Ok(dsp) => dsp,
            Err(error) => {
                unsafe { network.destroy(gl) };
                return Err(error);
            }
        };
        let zeros = vec![0.0_f32; STATE_SIZE];
        let mut buffers = Vec::with_capacity(4);
        let setup = (|| -> Result<()> {
            unsafe {
                let limit = gl.get_parameter_i64(glow::MAX_SHADER_STORAGE_BLOCK_SIZE);
                for size in [
                    weights.data.len() * 4,
                    STATE_SIZE * 4,
                    arena.len as usize * 4,
                ] {
                    if size as i64 > limit || size > i32::MAX as usize {
                        return Err(backend(format!(
                            "设备 SSBO 容量不足：需要 {size} 字节，支持 {limit} 字节"
                        )));
                    }
                }
                for slot in 0..4 {
                    let buffer = gl.create_buffer().map_err(backend)?;
                    buffers.push(buffer);
                    gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(buffer));
                    match slot {
                        0 => gl.buffer_data_u8_slice(
                            glow::SHADER_STORAGE_BUFFER,
                            bytemuck::cast_slice(&weights.data),
                            glow::STATIC_DRAW,
                        ),
                        1 | 2 => gl.buffer_data_u8_slice(
                            glow::SHADER_STORAGE_BUFFER,
                            bytemuck::cast_slice(&zeros),
                            glow::DYNAMIC_COPY,
                        ),
                        _ => gl.buffer_data_size(
                            glow::SHADER_STORAGE_BUFFER,
                            (arena.len * 4) as i32,
                            glow::DYNAMIC_COPY,
                        ),
                    }
                }
                let error = gl.get_error();
                if error != glow::NO_ERROR {
                    return Err(backend(format!(
                        "无法创建 GPU 缓冲区：OpenGL 0x{error:04x}"
                    )));
                }
            }
            Ok(())
        })();
        if let Err(error) = setup {
            unsafe {
                for buffer in buffers {
                    gl.delete_buffer(buffer);
                }
                dsp.destroy(gl);
                network.destroy(gl);
            }
            return Err(error);
        }
        Ok(Self {
            context,
            network,
            dsp,
            buffers: buffers.try_into().unwrap(),
            current: 0,
            input,
            output,
            zeros,
        })
    }

    pub(crate) fn frame(
        &mut self,
        input: &[f32; FRAME_SIZE],
        output: &mut [f32; FRAME_SIZE],
        attenuation: f32,
        reset: bool,
    ) -> Result<()> {
        let gl = &self.context.gl;
        unsafe {
            if reset {
                for buffer in &self.buffers[1..3] {
                    gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(*buffer));
                    gl.buffer_sub_data_u8_slice(
                        glow::SHADER_STORAGE_BUFFER,
                        0,
                        bytemuck::cast_slice(&self.zeros),
                    );
                }
                self.current = 0;
            }
            gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(self.buffers[0]));
            gl.bind_buffer_base(
                glow::SHADER_STORAGE_BUFFER,
                1,
                Some(self.buffers[1 + self.current]),
            );
            gl.bind_buffer_base(
                glow::SHADER_STORAGE_BUFFER,
                2,
                Some(self.buffers[1 + (self.current ^ 1)]),
            );
            gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 3, Some(self.buffers[3]));
            gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(self.buffers[3]));
            gl.buffer_sub_data_u8_slice(
                glow::SHADER_STORAGE_BUFFER,
                (self.input * 4) as i32,
                bytemuck::cast_slice(input),
            );
            self.dsp.prepare(gl);
            self.network.encode(gl);
            self.dsp.history(gl);
            self.network.decode(gl);
            self.dsp.finish(gl, attenuation);
            gl.memory_barrier(glow::BUFFER_UPDATE_BARRIER_BIT | glow::SHADER_STORAGE_BARRIER_BIT);
            let fence = gl
                .fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0)
                .map_err(backend)?;
            let mut flags = glow::SYNC_FLUSH_COMMANDS_BIT;
            loop {
                let status = gl.client_wait_sync(fence, flags, 1_000_000_000);
                flags = 0;
                if status == glow::ALREADY_SIGNALED || status == glow::CONDITION_SATISFIED {
                    break;
                }
                if status != glow::TIMEOUT_EXPIRED {
                    gl.delete_sync(fence);
                    return Err(backend("等待 GPU 输出失败；请重置或重新创建降噪器"));
                }
            }
            gl.delete_sync(fence);
            gl.get_buffer_sub_data(
                glow::SHADER_STORAGE_BUFFER,
                (self.output * 4) as i32,
                bytemuck::cast_slice_mut(output),
            );
            let error = gl.get_error();
            if error != glow::NO_ERROR {
                return Err(backend(format!(
                    "OpenGL 推理失败：0x{error:04x}；请重置或重新创建降噪器"
                )));
            }
            self.current ^= 1;
        }
        Ok(())
    }
}
impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let gl = &self.context.gl;
            self.network.destroy(gl);
            self.dsp.destroy(gl);
            for buffer in self.buffers {
                gl.delete_buffer(buffer);
            }
        }
    }
}
