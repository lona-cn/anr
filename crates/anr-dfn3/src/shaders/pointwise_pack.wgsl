// Standalone core WGSL: no f16 storage, cooperative operations, or enables.
// One initialization dispatch packs the 4,096 weights of a qualified 64 -> 64
// pointwise layer on the same GPU that will consume the compensated halves.
@group(0) @binding(0) var<storage, read> weights: array<f32>;
struct Params { values: array<vec4<u32>, 8>, }
@group(0) @binding(4) var<uniform> parameters: Params;
@group(0) @binding(5) var<storage, read_write> packed_weights: array<u32>;

fn p(i: u32) -> u32 { return parameters.values[i / 4u][i % 4u]; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i < 4096u {
        let value = weights[p(4u) + i];
        let high = quantizeToF16(value);
        // pack2x16float places high in bits 0..15 and the scaled residual in
        // bits 16..31, matching bitcast<vec2<f16>> in the matrix shader.
        packed_weights[p(21u) + i] = pack2x16float(vec2(high, (value - high) * 2048.0));
    }
}
