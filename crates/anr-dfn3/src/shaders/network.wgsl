@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> state_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> state_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;
struct Params { values: array<vec4<u32>, 8>, }
@group(0) @binding(4) var<uniform> parameters: Params;
var<workgroup> gru_input: array<f32, 256>;
var<workgroup> gru_old_state: array<f32, 256>;

fn p(i: u32) -> u32 { return parameters.values[i / 4u][i % 4u]; }
fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn activate(x: f32, kind: u32) -> f32 {
    if kind == 1u { return max(x, 0.0); }
    if kind == 2u { return sigmoid(x); }
    if kind == 3u { return tanh(x); }
    return x;
}

// Modes 0/1: ONNX Conv [Cout,Cin/groups,Kh,Kw] and ConvTranspose
// [Cin,Cout/groups,Kh,Kw]. Scratch activations use NCHW order.
fn convolution(index: u32, transposed: bool) {
    let ow = index % p(11u);
    let oh = (index / p(11u)) % p(10u);
    let oc = index / (p(11u) * p(10u));
    let inputs_per_group = p(6u) / p(18u);
    let outputs_per_group = p(9u) / p(18u);
    let group = oc / outputs_per_group;
    var sum = 0.0;
    if p(5u) != 0xffffffffu { sum = weights[p(5u) + oc]; }
    for (var ic = 0u; ic < inputs_per_group; ic += 1u) {
        let input_channel = group * inputs_per_group + ic;
        for (var kh = 0u; kh < p(12u); kh += 1u) {
            var iy: i32;
            if transposed {
                let t = i32(oh) + i32(p(16u)) - i32(kh);
                if t < 0 || t % i32(p(14u)) != 0 { continue; }
                iy = t / i32(p(14u));
            } else {
                iy = i32(oh * p(14u) + kh) - i32(p(16u));
            }
            if iy < 0 || iy >= i32(p(7u)) { continue; }
            for (var kw = 0u; kw < p(13u); kw += 1u) {
                var ix: i32;
                if transposed {
                    let t = i32(ow) + i32(p(17u)) - i32(kw);
                    if t < 0 || t % i32(p(15u)) != 0 { continue; }
                    ix = t / i32(p(15u));
                } else {
                    ix = i32(ow * p(15u) + kw) - i32(p(17u));
                }
                if ix < 0 || ix >= i32(p(8u)) { continue; }
                var wi: u32;
                if transposed {
                    wi = ((input_channel * outputs_per_group + oc % outputs_per_group)
                        * p(12u) + kh) * p(13u) + kw;
                } else {
                    wi = ((oc * inputs_per_group + ic) * p(12u) + kh) * p(13u) + kw;
                }
                let si = (input_channel * p(7u) + u32(iy)) * p(8u) + u32(ix);
                sum += scratch[p(2u) + si] * weights[p(4u) + wi];
            }
        }
    }
    scratch[p(3u) + index] = activate(sum, p(19u));
}

// Mode 2: exported Einsum weights retain [group,input,output] layout.
fn grouped_linear(index: u32) {
    let group = index / p(8u);
    let output_in_group = index % p(8u);
    var sum = 0.0;
    for (var i = 0u; i < p(7u); i += 1u) {
        let wi = (group * p(7u) + i) * p(8u) + output_in_group;
        sum += scratch[p(2u) + group * p(7u) + i] * weights[p(4u) + wi];
    }
    scratch[p(3u) + index] = activate(sum, p(19u));
}

fn reorder(index: u32) {
    var source: u32;
    if p(8u) == 0u {
        let channel = index % p(6u);
        let spatial = index / p(6u);
        source = channel * p(7u) + spatial;
    } else {
        let channel = index / p(7u);
        let spatial = index % p(7u);
        source = spatial * p(6u) + channel;
    }
    scratch[p(3u) + index] = scratch[p(2u) + source];
}

fn gru(index: u32, lane: u32) {
    // Every lane loads the entire shared vectors before any bounds return.
    for (var round = 0u; round < 4u; round += 1u) {
        let k = lane + round * 64u;
        gru_input[k] = scratch[p(2u) + k];
        gru_old_state[k] = state_in[p(8u) + k];
    }
    workgroupBarrier();
    if index >= p(1u) { return; }

    // Packed [k][Wz,Wr,Wh,Rz,Rr,Rh][j]; ONNX linear_before_reset=1.
    var wz = weights[p(10u) + index];
    var wr = weights[p(10u) + 256u + index];
    var wh = weights[p(10u) + 512u + index];
    var rz = weights[p(10u) + 768u + index];
    var rr = weights[p(10u) + 1024u + index];
    var rh = weights[p(10u) + 1280u + index];
    for (var k = 0u; k < 256u; k += 1u) {
        let base = p(4u) + k * 1536u + index;
        let x = gru_input[k];
        let old = gru_old_state[k];
        wz += x * weights[base];
        wr += x * weights[base + 256u];
        wh += x * weights[base + 512u];
        rz += old * weights[base + 768u];
        rr += old * weights[base + 1024u];
        rh += old * weights[base + 1280u];
    }
    let z = sigmoid(wz + rz);
    let r = sigmoid(wr + rr);
    let candidate = tanh(wh + r * rh);
    let value = (1.0 - z) * candidate + z * gru_old_state[index];
    scratch[p(3u) + index] = value;
    state_out[p(8u) + index] = value;
}

fn coefficients(index: u32) {
    // FC [96,10] + temporal Conv [10,1,96] -> DSP [5,96,2].
    let component = index % 2u;
    let frequency = (index / 2u) % 96u;
    let tap = index / 192u;
    let channel = tap * 2u + component;
    scratch[p(3u) + index] = scratch[p(2u) + frequency * 10u + channel]
        + scratch[p(4u) + channel * 96u + frequency];
}

fn network(index: u32, lane: u32) {
    // Uniform GRU branch precedes the per-lane bounds check for its barrier.
    if p(0u) == 5u { gru(index, lane); return; }
    if index >= p(1u) { return; }
    if p(0u) == 0u { convolution(index, false); }
    else if p(0u) == 1u { convolution(index, true); }
    else if p(0u) == 2u { grouped_linear(index); }
    else if p(0u) == 3u {
        scratch[p(3u) + index] = scratch[p(2u) + index] + scratch[p(4u) + index];
    }
    else if p(0u) == 4u { reorder(index); }
    else if p(0u) == 6u { coefficients(index); }
}
