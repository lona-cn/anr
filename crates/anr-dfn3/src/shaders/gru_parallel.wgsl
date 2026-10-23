// Append to network.wgsl. Parameters are the original mode-5 GRU offsets:
// p2=input scratch, p3=output scratch, p4=weights, p8=state, p10=bias.
// p24 is specialized to the K-lane width 16, 32 or 64. Each WG64 computes
// 64/p24 neurons, so 256 outputs require 64, 128 or 256 workgroups.
// Weights must be prepared as [Wz,Wr,Wh,Rz,Rr,Rh][neuron][k].
//
// Reuse the common gru_input/gru_old_state arrays: cooperative vector loads
// avoid repeating global input/state reads for every output and gate. They
// cost 2 KiB of workgroup memory; the six SoA partial planes add 1.5 KiB.
// No subgroup operations, subgroup-size assumptions or FP16 arithmetic.
var<workgroup> gru_partials: array<f32, 384>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let lane = local.x;
    let width = p(24u);
    let lane_k = lane % width;
    let neuron = group.x * (64u / width) + lane / width;
    for (var round = 0u; round < 4u; round += 1u) {
        let k = lane + round * 64u;
        gru_input[k] = scratch[p(2u) + k];
        gru_old_state[k] = state_in[p(8u) + k];
    }
    workgroupBarrier();

    // Bias is excluded from every partial, then added once by lane_k == 0.
    var wz = 0.0;
    var wr = 0.0;
    var wh = 0.0;
    var rz = 0.0;
    var rr = 0.0;
    var rh = 0.0;
    for (var k = lane_k; k < 256u; k += width) {
        let base = p(4u) + neuron * 256u + k;
        let x = gru_input[k];
        let old = gru_old_state[k];
        wz += x * weights[base];
        wr += x * weights[base + 65536u];
        wh += x * weights[base + 131072u];
        rz += old * weights[base + 196608u];
        rr += old * weights[base + 262144u];
        rh += old * weights[base + 327680u];
    }
    gru_partials[lane] = wz;
    gru_partials[64u + lane] = wr;
    gru_partials[128u + lane] = wh;
    gru_partials[192u + lane] = rz;
    gru_partials[256u + lane] = rr;
    gru_partials[320u + lane] = rh;
    workgroupBarrier();

    // Every invocation executes every barrier. The active K half never reads
    // another neuron's interval, including when four neurons share a WG.
    for (var step = width / 2u; step > 0u; step /= 2u) {
        if lane_k < step {
            gru_partials[lane] += gru_partials[lane + step];
            gru_partials[64u + lane] += gru_partials[64u + lane + step];
            gru_partials[128u + lane] += gru_partials[128u + lane + step];
            gru_partials[192u + lane] += gru_partials[192u + lane + step];
            gru_partials[256u + lane] += gru_partials[256u + lane + step];
            gru_partials[320u + lane] += gru_partials[320u + lane + step];
        }
        workgroupBarrier();
    }

    if lane_k == 0u {
        let bias = p(10u) + neuron;
        let sum_wz = gru_partials[lane] + weights[bias];
        let sum_wr = gru_partials[64u + lane] + weights[bias + 256u];
        let sum_wh = gru_partials[128u + lane] + weights[bias + 512u];
        let sum_rz = gru_partials[192u + lane] + weights[bias + 768u];
        let sum_rr = gru_partials[256u + lane] + weights[bias + 1024u];
        let sum_rh = gru_partials[320u + lane] + weights[bias + 1280u];
        // ONNX linear_before_reset=1: reset multiplies all of Rh, including
        // its recurrent bias, rather than the old state before the dot.
        let z = sigmoid(sum_wz + sum_rz);
        let r = sigmoid(sum_wr + sum_rr);
        let candidate = tanh(sum_wh + r * sum_rh);
        let value = (1.0 - z) * candidate + z * gru_old_state[neuron];
        scratch[p(3u) + neuron] = value;
        state_out[p(8u) + neuron] = value;
    }
}
