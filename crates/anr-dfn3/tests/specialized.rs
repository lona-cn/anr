#![cfg(feature = "spirv")]

use anr_dfn3::shaders::{self, NetworkKernel};
use spirv::{Capability, Op};

#[test]
fn fixed_vector_add_keeps_the_vulkan_1_1_capability_floor() {
    // Unused convolution/GRU dimensions are zero in a valid vector-add dispatch.
    // Specialization must not reject those unreachable paths or require FP16.
    let mut parameters = [0; 32];
    parameters[0] = 3;
    parameters[1] = 16;
    parameters[3] = 16;
    parameters[4] = 32;
    let words = shaders::network_spirv(&parameters, NetworkKernel::Scalar).unwrap();
    assert_core_capabilities(&words);
}

#[test]
fn parallel_gru_keeps_the_vulkan_1_1_capability_floor() {
    let mut parameters = [0; 32];
    parameters[0] = 5;
    parameters[1] = 256;
    parameters[3] = 256;
    parameters[10] = 6 * 256 * 256;
    for width in [16, 32, 64] {
        let words = shaders::network_spirv(&parameters, NetworkKernel::Gru(width)).unwrap();
        assert_core_capabilities(&words);
    }
}

#[test]
fn unsupported_kernel_geometry_is_rejected_before_execution() {
    for size in [0, 16, 128] {
        assert!(shaders::network_spirv(&[0; 32], NetworkKernel::Cooperative(size)).is_err());
    }
    for width in [0, 1, 15, 128] {
        assert!(shaders::network_spirv(&[0; 32], NetworkKernel::Gru(width)).is_err());
    }
}

fn assert_core_capabilities(words: &[u32]) {
    let mut offset = 5;
    let mut shader = false;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        assert!(count != 0 && offset + count <= words.len());
        if Op::from_u32(words[offset] & 0xffff) == Some(Op::Capability) {
            let capability = Capability::from_u32(words[offset + 1]);
            assert!(matches!(
                capability,
                Some(Capability::Shader | Capability::Matrix)
            ));
            shader |= capability == Some(Capability::Shader);
        }
        offset += count;
    }
    assert!(shader);
}
