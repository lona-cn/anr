//! Canonical compute kernels and one-time native backend compilation.

use crate::{Dispatch, ShaderKind};
#[cfg(any(feature = "glsl", feature = "spirv", feature = "hlsl"))]
use anr_core::Result;

mod specialized;

#[cfg(feature = "spirv")]
mod cooperative_spirv;

/// Canonical WGSL kernels for wgpu and the GLSL, SPIR-V, and HLSL translations.
pub fn wgsl(kind: ShaderKind) -> &'static str {
    match kind {
        ShaderKind::Network => concat!(
            include_str!("shaders/network.wgsl"),
            include_str!("shaders/network_main.wgsl"),
        ),
        ShaderKind::Dsp => include_str!("shaders/dsp.wgsl"),
        ShaderKind::Fft => include_str!("shaders/fft.wgsl"),
    }
}

/// Bake the fixed dispatch parameters into WGSL before backend compilation.
/// Only attenuation words 22 and 23 remain runtime uniform reads.
pub fn specialized_wgsl(dispatch: &Dispatch) -> String {
    specialized::wgsl(dispatch)
}

/// Native CUDA kernels shared by the CUDA and TensorRT executors.
pub fn cuda() -> &'static str {
    include_str!("shaders/cuda.cu")
}

#[cfg(feature = "spirv")]
fn validated(kind: ShaderKind) -> Result<(naga::Module, naga::valid::ModuleInfo)> {
    validated_source(kind, wgsl(kind))
}

#[cfg(any(feature = "glsl", feature = "spirv", feature = "hlsl"))]
fn validated_source(
    kind: ShaderKind,
    source: &str,
) -> Result<(naga::Module, naga::valid::ModuleInfo)> {
    let module = naga::front::wgsl::parse_str(source).map_err(|error| {
        crate::backend(format!(
            "{kind:?} WGSL parsing failed:\n{}",
            error.emit_to_string(source)
        ))
    })?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .map_err(|error| {
        crate::backend(format!(
            "{kind:?} WGSL validation failed:\n{}",
            error.emit_to_string(source)
        ))
    })?;
    Ok((module, info))
}

#[cfg(any(feature = "glsl", feature = "spirv"))]
fn unchecked() -> naga::proc::BoundsCheckPolicies {
    // The hash-locked model and its fixed schedule guarantee every buffer index.
    // In particular, D3D12 root SRVs/UAVs do not carry buffer-length metadata.
    naga::proc::BoundsCheckPolicies {
        index: naga::proc::BoundsCheckPolicy::Unchecked,
        buffer: naga::proc::BoundsCheckPolicy::Unchecked,
        image_load: naga::proc::BoundsCheckPolicy::Unchecked,
        binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
    }
}

/// Compile a fixed dispatch once when creating its desktop OpenGL program.
#[cfg(feature = "glsl")]
pub fn glsl(dispatch: &Dispatch) -> Result<String> {
    glsl_version(dispatch, naga::back::glsl::Version::Desktop(430))
}

/// Compile a fixed dispatch for native OpenGL ES 3.1, not WebGL.
#[cfg(feature = "glsl")]
pub fn glsl_es(dispatch: &Dispatch) -> Result<String> {
    glsl_version(dispatch, naga::back::glsl::Version::new_gles(310))
}

#[cfg(feature = "glsl")]
fn glsl_version(dispatch: &Dispatch, version: naga::back::glsl::Version) -> Result<String> {
    use naga::back::glsl as target;
    let kind = dispatch.shader;
    let (module, info) = validated_source(kind, &specialized_wgsl(dispatch))?;
    let options = target::Options {
        version,
        writer_flags: target::WriterFlags::empty(),
        binding_map: (0u8..5)
            .map(|binding| {
                (
                    naga::ResourceBinding {
                        group: 0,
                        binding: u32::from(binding),
                    },
                    binding,
                )
            })
            .collect(),
        // All shared values are explicitly initialized before use.
        zero_initialize_workgroup_memory: false,
    };
    let pipeline = target::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".into(),
        multiview: None,
    };
    let mut source = String::new();
    let mut writer = target::Writer::new(
        &mut source,
        &module,
        &info,
        &options,
        &pipeline,
        unchecked(),
    )
    .map_err(|error| {
        crate::backend(format!(
            "{kind:?} GLSL writer initialization failed: {error}"
        ))
    })?;
    writer
        .write()
        .map_err(|error| crate::backend(format!("{kind:?} GLSL compilation failed: {error}")))?;
    Ok(source)
}

/// Compile Vulkan 1.1-compatible SPIR-V 1.3 with set-zero bindings 0 through 4.
#[cfg(feature = "spirv")]
pub fn spirv(kind: ShaderKind) -> Result<Vec<u32>> {
    let (module, info) = validated(kind)?;
    write_spirv(&module, &info)
        .map_err(|error| crate::backend(format!("{kind:?} SPIR-V compilation failed: {error}")))
}

#[cfg(feature = "spirv")]
fn network_runtime_words(parameters: &[u32; 32]) -> u32 {
    let offsets = (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 21) | (1 << 22) | (1 << 23);
    if parameters[0] == 5 {
        offsets | (1 << 8) | (1 << 10)
    } else {
        offsets
    }
}

/// Reuse key for fixed Network shapes. Buffer offsets stay dynamic, allowing
/// repeated layers and all five GRUs to share their compiled pipelines.
#[cfg(feature = "spirv")]
pub fn network_key(parameters: &[u32; 32]) -> [u32; 32] {
    let mut key = *parameters;
    let dynamic = network_runtime_words(parameters);
    for (index, value) in key.iter_mut().enumerate() {
        if dynamic & (1 << index) != 0 {
            *value = 0;
        }
    }
    key
}

/// Vulkan implementation chosen before recording a Network dispatch.
#[cfg(feature = "spirv")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NetworkKernel {
    Scalar,
    Cooperative(u32),
    Gru(u32),
}

/// Compile fixed Network modes/shapes once, retaining runtime buffer offsets.
/// Per-frame attenuation never causes pipeline compilation.
#[cfg(feature = "spirv")]
pub fn network_spirv(parameters: &[u32; 32], kernel: NetworkKernel) -> Result<Vec<u32>> {
    let cooperative = matches!(kernel, NetworkKernel::Cooperative(_));
    let source = match kernel {
        NetworkKernel::Scalar => wgsl(ShaderKind::Network),
        NetworkKernel::Cooperative(32 | 64) => concat!(
            "enable f16;\nenable wgpu_cooperative_matrix;\n",
            include_str!("shaders/network.wgsl"),
            include_str!("shaders/cooperative_pointwise_packed.wgsl"),
        ),
        NetworkKernel::Gru(16 | 32 | 64) => concat!(
            include_str!("shaders/network.wgsl"),
            include_str!("shaders/gru_parallel.wgsl"),
        ),
        _ => return Err(crate::backend("Unsupported Vulkan Network kernel geometry")),
    };
    let mut parameters = *parameters;
    match kernel {
        NetworkKernel::Cooperative(size) => parameters[20] = size,
        NetworkKernel::Gru(width) => parameters[24] = width,
        NetworkKernel::Scalar => {}
    }
    let source =
        specialized::source_with_dynamic(source, &parameters, network_runtime_words(&parameters));
    let module = naga::front::wgsl::parse_str(&source).map_err(|error| {
        crate::backend(format!(
            "Network WGSL parsing failed:\n{}",
            error.emit_to_string(&source)
        ))
    })?;
    let capabilities = if cooperative {
        naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::SHADER_FLOAT16
            | naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32
            | naga::valid::Capabilities::SUBGROUP
    } else {
        naga::valid::Capabilities::empty()
    };
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
        .subgroup_stages(naga::valid::ShaderStages::COMPUTE)
        .subgroup_operations(naga::valid::SubgroupOperationSet::VOTE)
        .validate(&module)
        .map_err(|error| crate::backend(format!("Fixed Network validation failed: {error:?}")))?;
    let words = write_spirv(&module, &info).map_err(|error| {
        crate::backend(format!("Fixed Network SPIR-V compilation failed: {error}"))
    })?;
    if cooperative {
        cooperative_spirv::publish_workgroup_memory(&words)
    } else {
        Ok(words)
    }
}

/// Pack immutable pointwise weights once without requiring FP16 shader types.
#[cfg(feature = "spirv")]
pub fn pointwise_pack_spirv() -> Result<Vec<u32>> {
    let source = include_str!("shaders/pointwise_pack.wgsl");
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|error| crate::backend(format!("Pointwise pack parsing failed: {error}")))?;
    // QuantizeToF16 is a core FP32 SPIR-V operation, not shaderFloat16.
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
    )
    .validate(&module)
    .map_err(|error| crate::backend(format!("Pointwise pack validation failed: {error:?}")))?;
    write_spirv(&module, &info)
        .map_err(|error| crate::backend(format!("Pointwise packing compilation failed: {error}")))
}

#[cfg(feature = "spirv")]
fn write_spirv(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
) -> std::result::Result<Vec<u32>, naga::back::spv::Error> {
    use naga::back::spv as target;
    let options = target::Options {
        lang_version: (1, 3),
        flags: target::WriterFlags::empty(),
        fake_missing_bindings: false,
        binding_map: (0..6)
            .map(|binding| {
                (
                    naga::ResourceBinding { group: 0, binding },
                    target::BindingInfo {
                        descriptor_set: 0,
                        binding,
                        binding_array_size: None,
                    },
                )
            })
            .collect(),
        bounds_check_policies: unchecked(),
        zero_initialize_workgroup_memory: target::ZeroInitializeWorkgroupMemoryMode::None,
        force_loop_bounding: false,
        emit_int_div_checks: false,
        use_storage_input_output_16: false,
        ..Default::default()
    };
    let pipeline = target::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".into(),
    };
    target::write_vec(module, info, &options, Some(&pipeline))
}

/// Generated HLSL plus the real (possibly renamed) entry point for D3DCompile.
#[cfg(feature = "hlsl")]
#[derive(Debug)]
pub struct Hlsl {
    pub source: String,
    pub entry_point: String,
}

/// Compile a fixed dispatch to SM5.1 HLSL for D3D12 root descriptors.
#[cfg(feature = "hlsl")]
pub fn hlsl(dispatch: &Dispatch) -> Result<Hlsl> {
    use naga::back::hlsl as target;
    let kind = dispatch.shader;
    let (module, info) = validated_source(kind, &specialized_wgsl(dispatch))?;
    let options = target::Options {
        shader_model: target::ShaderModel::V5_1,
        binding_map: (0..5)
            .map(|binding| {
                (
                    naga::ResourceBinding { group: 0, binding },
                    target::BindTarget {
                        space: 0,
                        register: binding,
                        binding_array_size: None,
                        dynamic_storage_buffer_offsets_index: None,
                        restrict_indexing: false,
                    },
                )
            })
            .collect(),
        fake_missing_bindings: false,
        // Suppress runtime-array GetDimensions: root buffers have no length.
        restrict_indexing: false,
        zero_initialize_workgroup_memory: false,
        force_loop_bounding: false,
        ..Default::default()
    };
    let pipeline = target::PipelineOptions {
        entry_point: Some((naga::ShaderStage::Compute, "main".into())),
    };
    let mut source = String::new();
    let reflection = target::Writer::new(&mut source, &options, &pipeline)
        .write(&module, &info, None)
        .map_err(|error| crate::backend(format!("{kind:?} HLSL compilation failed: {error}")))?;
    let entry_point = reflection
        .entry_point_names
        .into_iter()
        .next()
        .ok_or_else(|| crate::backend(format!("{kind:?} HLSL entry point was not emitted")))?
        .map_err(|error| crate::backend(format!("{kind:?} HLSL entry point failed: {error}")))?;
    Ok(Hlsl {
        source,
        entry_point,
    })
}
