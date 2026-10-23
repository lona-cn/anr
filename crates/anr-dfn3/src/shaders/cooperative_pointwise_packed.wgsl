// Append to network.wgsl with f16 and wgpu_cooperative_matrix enabled.
// Only qualified 64 -> 64 pointwise ReLU convolutions use this entry point.
// The packing shader writes the same four-byte element as a u32. Loading
// vec2<f16> here avoids per-frame unpacking/conversion and preserves both halves.
@group(0) @binding(5) var<storage, read> packed_weights: array<vec2<f16>>;

// One [16,64] activation tile is shared by the entire 64-thread workgroup.
// SG32 uses two adjacent channel tiles at the same spatial origin; SG64 uses
// one. Each subgroup has its own [64,16] weight and [16,16] result tiles.
// These declarations total 16,640 bytes: 4,096 input + 8,192 weight +
// 4,096 result + 256 range flags (excluding the common prefix's GRU arrays).
var<workgroup> input_hi: array<f16, 1024>;
var<workgroup> input_lo: array<f16, 1024>;
var<workgroup> weight_hi: array<f16, 2048>;
var<workgroup> weight_lo: array<f16, 2048>;
var<workgroup> result_hi: array<f32, 512>;
var<workgroup> result_lo: array<f32, 512>;
var<workgroup> input_out_of_range: array<u32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) group: vec3<u32>,
        @builtin(subgroup_id) subgroup: u32,
        @builtin(subgroup_invocation_id) lane: u32) {
    let subgroup_size = p(20u);
    let first_tile = group.x * (64u / subgroup_size);
    let spatial_start = (first_tile / 4u) * 16u;
    let channel_start = ((first_tile + subgroup) % 4u) * 16u;
    let tensor_base = subgroup * 1024u;
    let result_base = subgroup * 256u;
    let spatial = p(10u) * p(11u);

    var needs_scalar = false;
    // Local IDs, not subgroup lanes, partition this single shared input tile.
    for (var i = local_id.x; i < 1024u; i += 64u) {
        let position = spatial_start + i % 16u;
        let input_channel = i / 16u;
        var value = 0.0;
        if position < spatial {
            value = scratch[p(2u) + input_channel * spatial + position];
        }
        let in_half_range = abs(value) <= 65504.0;
        needs_scalar = needs_scalar || !in_half_range;
        if !in_half_range { value = 0.0; }
        let high = quantizeToF16(value);
        input_hi[i] = f16(high);
        input_lo[i] = f16((value - high) * 2048.0);
    }
    input_out_of_range[local_id.x] = select(0u, 1u, needs_scalar);

    for (var i = lane; i < 1024u; i += subgroup_size) {
        let output_channel = channel_start + i / 64u;
        let halves = packed_weights[p(21u) + output_channel * 64u + i % 64u];
        weight_hi[tensor_base + i] = halves.x;
        weight_lo[tensor_base + i] = halves.y;
    }
    for (var i = lane; i < 256u; i += subgroup_size) {
        result_hi[result_base + i] = 0.0;
    }
    workgroupBarrier();

    // Every subgroup must see ALL 64 writers' flags, not merely those of its
    // own lanes. Both SG32 votes execute separately before the OR: no short
    // circuit and no assumption that the two subgroups execute in lockstep.
    let first_range_vote = subgroupAny(input_out_of_range[lane] != 0u);
    var scalar_fallback = first_range_vote;
    if subgroup_size == 32u {
        let second_range_vote = subgroupAny(input_out_of_range[lane + 32u] != 0u);
        scalar_fallback = first_range_vote || second_range_vote;
    }

    // Matrix operations and both workgroup barriers are unconditional, even
    // when a subgroup will discard these results and use FP32 convolution.
    var high = coopLoad<coop_mat16x16<f32, C>>(&result_hi[result_base], 16u);
    var low = high;
    for (var k = 0u; k < 64u; k += 16u) {
        let a_high = coopLoad<coop_mat16x16<f16, A>>(&input_hi[k * 16u], 16u);
        let a_low = coopLoad<coop_mat16x16<f16, A>>(&input_lo[k * 16u], 16u);
        let b_high = coopLoad<coop_mat16x16<f16, B>>(&weight_hi[tensor_base + k], 64u);
        let b_low = coopLoad<coop_mat16x16<f16, B>>(&weight_lo[tensor_base + k], 64u);
        high = coopMultiplyAdd(a_high, b_high, high);
        low = coopMultiplyAdd(a_high, b_low, low);
        low = coopMultiplyAdd(a_low, b_high, low);
    }
    coopStore(high, &result_hi[result_base], 16u);
    coopStore(low, &result_lo[result_base], 16u);
    workgroupBarrier();

    for (var i = lane; i < 256u; i += subgroup_size) {
        let position = spatial_start + i % 16u;
        // Spatial tails (including S=8) are zero-filled above and never stored.
        if position < spatial {
            let channel = channel_start + i / 16u;
            if scalar_fallback {
                // Keep the original FP32 weights at p(4) for this path.
                convolution(channel * spatial + position, false);
            } else {
                var value = result_hi[result_base + i]
                    + result_lo[result_base + i] * (1.0 / 2048.0);
                if p(5u) != 0xffffffffu { value += weights[p(5u) + channel]; }
                scratch[p(3u) + channel * spatial + position] = max(value, 0.0);
            }
        }
    }
}
