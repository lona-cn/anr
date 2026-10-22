use crate::context::GlContext;
use crate::{FRAME_SIZE, GlApi, STATE_SIZE, backend};
use anr_core::Result;
use anr_dfn3::{Executor, Model, PARAMETER_BYTES};
use glow::HasContext;

struct Kernel {
    program: glow::NativeProgram,
}
impl Kernel {
    unsafe fn new(gl: &glow::Context, source: &str) -> Result<Self> {
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
            Ok(Self { program })
        }
    }
}

pub(crate) struct Gpu<const ES: bool> {
    context: GlContext,
    programs: Vec<Kernel>,
    buffers: [glow::NativeBuffer; 5],
    model: Model,
    stride: u32,
    current: usize,
    zeros: Vec<f32>,
}
impl<const ES: bool> Gpu<ES> {
    pub(crate) fn new(mut model: Model) -> Result<Self> {
        let context = GlContext::new(if ES { GlApi::Gles } else { GlApi::Desktop })?;
        let gl = &context.gl;
        let alignment = unsafe { gl.get_parameter_i32(glow::UNIFORM_BUFFER_OFFSET_ALIGNMENT) };
        if alignment <= 0 {
            return Err(backend("驱动返回了无效的 uniform buffer 对齐值"));
        }
        let stride = PARAMETER_BYTES.div_ceil(alignment as u32) * alignment as u32;
        let parameters = model.parameter_bytes(stride);
        let zeros = vec![0.0_f32; STATE_SIZE];
        let mut programs = Vec::with_capacity(model.dispatches.len());
        let mut buffers = Vec::with_capacity(5);
        let setup = (|| -> Result<()> {
            unsafe {
                for dispatch in &model.dispatches {
                    let source = if ES {
                        anr_dfn3::shaders::glsl_es(dispatch)?
                    } else {
                        anr_dfn3::shaders::glsl(dispatch)?
                    };
                    programs.push(Kernel::new(gl, &source)?);
                }
                let limit = gl.get_parameter_i64(glow::MAX_SHADER_STORAGE_BLOCK_SIZE);
                for size in [
                    model.weights.len() * 4,
                    STATE_SIZE * 4,
                    model.scratch_len as usize * 4,
                ] {
                    if size as i64 > limit || size > i32::MAX as usize {
                        return Err(backend(format!(
                            "设备 SSBO 容量不足：需要 {size} 字节，支持 {limit} 字节"
                        )));
                    }
                }
                for slot in 0..5 {
                    let buffer = gl.create_buffer().map_err(backend)?;
                    buffers.push(buffer);
                    let target = if slot == 4 {
                        glow::UNIFORM_BUFFER
                    } else {
                        glow::SHADER_STORAGE_BUFFER
                    };
                    gl.bind_buffer(target, Some(buffer));
                    match slot {
                        0 => gl.buffer_data_u8_slice(
                            target,
                            bytemuck::cast_slice(&model.weights),
                            glow::STATIC_DRAW,
                        ),
                        1 | 2 => gl.buffer_data_u8_slice(
                            target,
                            bytemuck::cast_slice(&zeros),
                            glow::DYNAMIC_COPY,
                        ),
                        3 => gl.buffer_data_size(
                            target,
                            (model.scratch_len * 4) as i32,
                            glow::DYNAMIC_COPY,
                        ),
                        _ => gl.buffer_data_u8_slice(target, &parameters, glow::DYNAMIC_DRAW),
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
                for kernel in programs {
                    gl.delete_program(kernel.program);
                }
            }
            return Err(error);
        }
        model.weights = Vec::new();
        Ok(Self {
            context,
            programs,
            buffers: buffers.try_into().unwrap(),
            model,
            stride,
            current: 0,
            zeros,
        })
    }
}
impl<const ES: bool> Executor for Gpu<ES> {
    fn frame(
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
            if self.model.update_attenuation(attenuation) {
                let step = self.model.attenuation_dispatch;
                let offset = step as u32 * self.stride + 22 * 4;
                gl.bind_buffer(glow::UNIFORM_BUFFER, Some(self.buffers[4]));
                gl.buffer_sub_data_u8_slice(
                    glow::UNIFORM_BUFFER,
                    offset as i32,
                    bytemuck::cast_slice(&self.model.dispatches[step].params[22..24]),
                );
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
                (self.model.input * 4) as i32,
                bytemuck::cast_slice(input),
            );
            for (index, dispatch) in self.model.dispatches.iter().enumerate() {
                gl.use_program(Some(self.programs[index].program));
                gl.bind_buffer_range(
                    glow::UNIFORM_BUFFER,
                    4,
                    Some(self.buffers[4]),
                    (index as u32 * self.stride) as i32,
                    PARAMETER_BYTES as i32,
                );
                gl.dispatch_compute(dispatch.groups, 1, 1);
                gl.memory_barrier(glow::SHADER_STORAGE_BARRIER_BIT);
            }
            gl.memory_barrier(glow::BUFFER_UPDATE_BARRIER_BIT | glow::SHADER_STORAGE_BARRIER_BIT);
            let fence = gl
                .fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0)
                .map_err(backend)?;
            let mut flags = glow::SYNC_FLUSH_COMMANDS_BIT;
            loop {
                // Native ES glClientWaitSync accepts nanoseconds, unlike
                // glWaitSync's TIMEOUT_IGNORED rule or WebGL's timeout cap.
                // glow 0.16 takes i32 and casts to u64, so keep it positive.
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
            if ES {
                // GLES has no glGetBufferSubData. Only the completed frame's
                // 480 floats are mapped, not weights, state or the full arena.
                let mapped = gl.map_buffer_range(
                    glow::SHADER_STORAGE_BUFFER,
                    (self.model.output * 4) as i32,
                    std::mem::size_of_val(output) as i32,
                    glow::MAP_READ_BIT,
                );
                if mapped.is_null() {
                    let error = gl.get_error();
                    return Err(backend(format!(
                        "映射 GLES 输出失败：0x{error:04x}；请重置或重新创建降噪器"
                    )));
                }
                // Copy bytes: the native mapped address need not be aligned
                // for Rust f32 references. No fallible operation precedes unmap.
                std::ptr::copy_nonoverlapping(
                    mapped,
                    output.as_mut_ptr().cast::<u8>(),
                    std::mem::size_of_val(output),
                );
                if !self.context.unmap_output() {
                    return Err(backend(
                        "解除 GLES 输出映射失败或映射内容已损坏；请重置或重新创建降噪器",
                    ));
                }
            } else {
                gl.get_buffer_sub_data(
                    glow::SHADER_STORAGE_BUFFER,
                    (self.model.output * 4) as i32,
                    bytemuck::cast_slice_mut(output),
                );
            }
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
impl<const ES: bool> Drop for Gpu<ES> {
    fn drop(&mut self) {
        unsafe {
            let gl = &self.context.gl;
            for kernel in &self.programs {
                gl.delete_program(kernel.program);
            }
            for buffer in self.buffers {
                gl.delete_buffer(buffer);
            }
        }
    }
}
