//! Canonical compute kernels and one-time native backend compilation.

use crate::ShaderKind;
#[cfg(any(feature = "glsl", feature = "spirv", feature = "hlsl"))]
use anr_core::Result;

/// Canonical WGSL kernels for wgpu and the GLSL, SPIR-V, and HLSL translations.
pub fn wgsl(kind: ShaderKind) -> &'static str {
    match kind {
        ShaderKind::Network => include_str!("shaders/network.wgsl"),
        ShaderKind::Dsp => include_str!("shaders/dsp.wgsl"),
        ShaderKind::Fft => include_str!("shaders/fft.wgsl"),
    }
}

/// Native CUDA kernels shared by the CUDA and TensorRT executors.
pub fn cuda() -> &'static str {
    include_str!("shaders/cuda.cu")
}

#[cfg(any(feature = "glsl", feature = "spirv", feature = "hlsl"))]
fn validated(kind: ShaderKind) -> Result<(naga::Module, naga::valid::ModuleInfo)> {
    let source = wgsl(kind);
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

/// Compile once when creating the OpenGL program, never during frame execution.
#[cfg(feature = "glsl")]
pub fn glsl(kind: ShaderKind) -> Result<String> {
    glsl_version(kind, naga::back::glsl::Version::Desktop(430))
}

/// Compile native OpenGL ES 3.1 shaders, not WebGL shaders.
#[cfg(feature = "glsl")]
pub fn glsl_es(kind: ShaderKind) -> Result<String> {
    glsl_version(kind, naga::back::glsl::Version::new_gles(310))
}

#[cfg(feature = "glsl")]
fn glsl_version(kind: ShaderKind, version: naga::back::glsl::Version) -> Result<String> {
    use naga::back::glsl as target;
    let (module, info) = validated(kind)?;
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
    use naga::back::spv as target;
    let (module, info) = validated(kind)?;
    let options = target::Options {
        lang_version: (1, 3),
        flags: target::WriterFlags::empty(),
        fake_missing_bindings: false,
        binding_map: (0..5)
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
        ..Default::default()
    };
    let pipeline = target::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".into(),
    };
    target::write_vec(&module, &info, &options, Some(&pipeline))
        .map_err(|error| crate::backend(format!("{kind:?} SPIR-V compilation failed: {error}")))
}

/// Generated HLSL plus the real (possibly renamed) entry point for D3DCompile.
#[cfg(feature = "hlsl")]
#[derive(Debug)]
pub struct Hlsl {
    pub source: String,
    pub entry_point: String,
}

/// Compile SM5.1 HLSL suitable for D3D12 root SRV/UAV/CBV descriptors.
#[cfg(feature = "hlsl")]
pub fn hlsl(kind: ShaderKind) -> Result<Hlsl> {
    use naga::back::hlsl as target;
    let (module, info) = validated(kind)?;
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
