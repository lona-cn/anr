//! Naga 30 emits the Vulkan memory model for cooperative matrices, but its
//! workgroup loads/stores and barriers still use GLSL450 memory semantics.
//! Publish shared writes explicitly, including those in the unchanged GRU.

use anr_core::Result;
use spirv::{MemoryAccess, MemorySemantics, Op, StorageClass};

#[derive(Clone, Copy)]
enum Id {
    Other,
    WorkgroupPointerType,
    WorkgroupPointer,
    Constant { ty: u32, value: u32, published: u32 },
}

fn memory_operand(instruction: &[u32], ids: &[Id]) -> Option<usize> {
    let (pointer, mask) = match Op::from_u32(instruction[0] & 0xffff) {
        Some(Op::Load) => (3, 4),
        Some(Op::Store) => (1, 3),
        Some(Op::CooperativeMatrixLoadKHR) => (3, 6),
        Some(Op::CooperativeMatrixStoreKHR) => (1, 5),
        _ => return None,
    };
    matches!(ids[instruction[pointer] as usize], Id::WorkgroupPointer).then_some(mask)
}

pub(super) fn publish_workgroup_memory(words: &[u32]) -> Result<Vec<u32>> {
    let mut ids = vec![Id::Other; words[3] as usize];
    let mut bound = words[3];
    let mut extra = 0;
    let mut offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        if count == 0 || offset + count > words.len() {
            return Err(crate::backend(
                "invalid cooperative SPIR-V instruction length",
            ));
        }
        let instruction = &words[offset..offset + count];
        match Op::from_u32(instruction[0] & 0xffff) {
            Some(Op::TypePointer) if instruction[2] == StorageClass::Workgroup as u32 => {
                ids[instruction[1] as usize] = Id::WorkgroupPointerType;
            }
            Some(
                Op::Variable
                | Op::FunctionParameter
                | Op::AccessChain
                | Op::InBoundsAccessChain
                | Op::PtrAccessChain
                | Op::InBoundsPtrAccessChain
                | Op::CopyObject
                | Op::CopyLogical
                | Op::Phi
                | Op::Select
                | Op::FunctionCall
                | Op::Load
                | Op::Bitcast,
            ) if matches!(ids[instruction[1] as usize], Id::WorkgroupPointerType) => {
                ids[instruction[2] as usize] = Id::WorkgroupPointer;
            }
            Some(Op::Constant) if count == 4 => {
                ids[instruction[2] as usize] = Id::Constant {
                    ty: instruction[1],
                    value: instruction[3],
                    published: 0,
                };
            }
            Some(Op::ControlBarrier) => {
                let semantics = instruction[3] as usize;
                let Id::Constant {
                    ty,
                    value,
                    published,
                } = ids[semantics]
                else {
                    return Err(crate::backend(
                        "cooperative barrier semantics must be constant",
                    ));
                };
                let flags = MemorySemantics::from_bits_retain(value);
                let visibility = MemorySemantics::MAKE_AVAILABLE | MemorySemantics::MAKE_VISIBLE;
                if flags.contains(MemorySemantics::WORKGROUP_MEMORY)
                    && !flags.contains(visibility)
                    && published == 0
                {
                    // Do not change the pooled original constant: it may also
                    // be used as an ordinary integer elsewhere in the module.
                    ids[semantics] = Id::Constant {
                        ty,
                        value,
                        published: bound,
                    };
                    bound += 1;
                    extra += 4;
                }
            }
            _ => {}
        }
        if let Some(mask) = memory_operand(instruction, &ids) {
            if count < mask {
                return Err(crate::backend(
                    "cooperative matrix memory stride is missing",
                ));
            }
            extra += usize::from(count == mask);
        }
        offset += count;
    }

    let mut output = Vec::with_capacity(words.len() + extra);
    output.extend_from_slice(&words[..5]);
    output[3] = bound;
    let mut declared = false;
    offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        let instruction = &words[offset..offset + count];
        let op = Op::from_u32(instruction[0] & 0xffff);
        if op == Some(Op::Function) && !declared {
            // Types and constants must precede the first function definition.
            for id in &ids {
                if let Id::Constant {
                    ty,
                    value,
                    published,
                } = *id
                    && published != 0
                {
                    output.extend_from_slice(&[
                        (4 << 16) | Op::Constant as u32,
                        ty,
                        published,
                        value
                            | (MemorySemantics::MAKE_AVAILABLE | MemorySemantics::MAKE_VISIBLE)
                                .bits(),
                    ]);
                }
            }
            declared = true;
        }
        let start = output.len();
        output.extend_from_slice(instruction);
        if let Some(mask) = memory_operand(instruction, &ids) {
            if mask == count {
                output[start] += 1 << 16;
                output.push(MemoryAccess::NON_PRIVATE_POINTER.bits());
            } else {
                output[start + mask] |= MemoryAccess::NON_PRIVATE_POINTER.bits();
            }
        }
        if op == Some(Op::ControlBarrier)
            && let Id::Constant { published, .. } = ids[instruction[3] as usize]
            && published != 0
        {
            output[start + 3] = published;
        }
        offset += count;
    }
    Ok(output)
}
