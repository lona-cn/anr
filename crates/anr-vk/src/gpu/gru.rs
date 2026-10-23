use anr_dfn3::{Model, ShaderKind};

/// Replaces each GRU's [k][six gates][j] block with [gate][j][k].
/// Call exactly once, before uploading weights for the parallel GRU kernels.
/// Biases, offsets and the total GPU weight-buffer size are unchanged; one
/// temporary matrix block is reused across all five recurrent layers.
pub(super) fn prepare(model: &mut Model) {
    const HIDDEN: usize = 256;
    const GATES: usize = 6;
    const BLOCK: usize = GATES * HIDDEN * HIDDEN;
    let mut transposed = Vec::with_capacity(BLOCK);
    for dispatch in &model.dispatches {
        if dispatch.shader != ShaderKind::Network || dispatch.params[0] != 5 {
            continue;
        }
        let start = dispatch.params[4] as usize;
        let block = &mut model.weights[start..start + BLOCK];
        transposed.clear();
        for gate in 0..GATES {
            for j in 0..HIDDEN {
                for k in 0..HIDDEN {
                    transposed.push(block[(k * GATES + gate) * HIDDEN + j]);
                }
            }
        }
        block.copy_from_slice(&transposed);
    }
}

/// One WG64 covers 64 / width outputs: widths 16/32/64 require 64/128/256
/// workgroups respectively. The same width must be specialized into p(24).
pub(super) fn groups(width: u32) -> u32 {
    assert!(matches!(width, 16 | 32 | 64));
    256 / (64 / width)
}
