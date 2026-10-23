#![cfg(feature = "spirv")]

use anr_dfn3::{ShaderKind, shaders};
use spirv::{Capability, MemoryAccess, MemorySemantics, Op, StorageClass};
use std::collections::{HashMap, HashSet};

fn instructions(words: &[u32]) -> Vec<&[u32]> {
    let mut offset = 5;
    let mut instructions = Vec::new();
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        assert!(count != 0 && offset + count <= words.len());
        instructions.push(&words[offset..offset + count]);
        offset += count;
    }
    instructions
}

fn op(instruction: &[u32]) -> Option<Op> {
    Op::from_u32(instruction[0] & 0xffff)
}

#[test]
fn scalar_kernels_do_not_require_optional_device_features() {
    for kind in ShaderKind::ALL {
        let words = shaders::spirv(kind).unwrap();
        for instruction in instructions(&words) {
            if op(instruction) == Some(Op::Capability) {
                assert!(
                    matches!(
                        Capability::from_u32(instruction[1]),
                        Some(Capability::Shader | Capability::Matrix)
                    ),
                    "{kind:?} requires an optional capability: {}",
                    instruction[1]
                );
            }
        }
    }
}

#[test]
fn pointwise_weight_packing_uses_only_core_shader_capabilities() {
    let words = shaders::pointwise_pack_spirv().unwrap();
    let mut shader = false;
    for instruction in instructions(&words) {
        if op(instruction) == Some(Op::Capability) {
            let capability = Capability::from_u32(instruction[1]);
            assert!(matches!(
                capability,
                Some(Capability::Shader | Capability::Matrix)
            ));
            shader |= capability == Some(Capability::Shader);
        }
    }
    assert!(shader);
}

#[test]
fn cooperative_kernel_does_not_require_fp16_stage_io() {
    let words = shaders::network_spirv(&[0; 32], shaders::NetworkKernel::Cooperative(32)).unwrap();
    for instruction in instructions(&words) {
        if op(instruction) == Some(Op::Capability) {
            assert_ne!(instruction[1], Capability::StorageInputOutput16 as u32);
        }
    }
}

#[test]
fn cooperative_workgroup_reads_and_writes_are_visible_between_invocations() {
    let words = shaders::network_spirv(&[0; 32], shaders::NetworkKernel::Cooperative(32)).unwrap();
    let instructions = instructions(&words);
    let mut workgroup_types = HashSet::new();
    let mut workgroup_pointers = HashSet::new();
    let mut constants = HashMap::new();
    for instruction in &instructions {
        match op(instruction) {
            Some(Op::TypePointer) if instruction[2] == StorageClass::Workgroup as u32 => {
                workgroup_types.insert(instruction[1]);
            }
            Some(
                Op::Variable
                | Op::AccessChain
                | Op::InBoundsAccessChain
                | Op::CopyObject
                | Op::FunctionParameter,
            ) if workgroup_types.contains(&instruction[1]) => {
                workgroup_pointers.insert(instruction[2]);
            }
            Some(Op::Constant) if instruction.len() == 4 => {
                constants.insert(instruction[2], instruction[3]);
            }
            _ => {}
        }
    }
    for instruction in &instructions {
        let pointer_and_mask = match op(instruction) {
            Some(Op::Load) => Some((3, 4)),
            Some(Op::Store) => Some((1, 3)),
            Some(Op::CooperativeMatrixLoadKHR) => Some((3, 6)),
            Some(Op::CooperativeMatrixStoreKHR) => Some((1, 5)),
            _ => None,
        };
        if let Some((pointer, mask)) = pointer_and_mask
            && workgroup_pointers.contains(&instruction[pointer])
        {
            assert!(
                instruction.len() > mask,
                "private workgroup {:?}",
                op(instruction)
            );
            assert_ne!(
                instruction[mask] & MemoryAccess::NON_PRIVATE_POINTER.bits(),
                0
            );
        }
        if op(instruction) == Some(Op::ControlBarrier) {
            let semantics = MemorySemantics::from_bits_retain(constants[&instruction[3]]);
            if semantics.contains(MemorySemantics::WORKGROUP_MEMORY) {
                assert!(
                    semantics
                        .contains(MemorySemantics::MAKE_AVAILABLE | MemorySemantics::MAKE_VISIBLE)
                );
            }
        }
    }
}
