@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> state_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> state_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;
struct Params { values: array<vec4<u32>, 8>, }
@group(0) @binding(4) var<uniform> parameters: Params;
var<workgroup> nonzero_mean: atomic<u32>;
var<workgroup> nonzero_norm: atomic<u32>;

fn p(i: u32) -> u32 { return parameters.values[i / 4u][i % 4u]; }
fn spectrum(f: u32) -> vec2<f32> {
    return vec2<f32>(scratch[p(6u) + 2u * f], scratch[p(6u) + 2u * f + 1u]);
}

fn features(i: u32) {
    if i == 0u {
        atomicStore(&nonzero_mean, 0u);
        atomicStore(&nonzero_norm, 0u);
    }
    workgroupBarrier();
    if i < 32u && state_in[i] != 0.0 { atomicOr(&nonzero_mean, 1u); }
    if i < 96u && state_in[32u + i] != 0.0 { atomicOr(&nonzero_norm, 1u); }
    workgroupBarrier();
    let beta = 0.009999990463256836;
    if i < 32u {
        let start = u32(weights[p(19u) + i]);
        let end = u32(weights[p(19u) + 32u + i]);
        var power = 0.0;
        for (var f = start; f < end; f += 1u) {
            let z = spectrum(f);
            power += (z.x * z.x + z.y * z.y) * weights[p(20u) + f];
        }
        let energy = 10.0 * (log(power + 1.000000013351432e-10) / 2.3025851249694824);
        var old = state_in[i];
        if atomicLoad(&nonzero_mean) == 0u { old = weights[p(17u) + i]; }
        let mean = old - (old - energy) * beta;
        state_out[i] = mean;
        scratch[p(7u) + i] = (energy - mean) / 40.0;
    }
    if i < 96u {
        let z = spectrum(i);
        let magnitude = sqrt(z.x * z.x + z.y * z.y);
        var old = state_in[32u + i];
        if atomicLoad(&nonzero_norm) == 0u { old = weights[p(18u) + i]; }
        let norm = old - (old - magnitude) * beta;
        state_out[32u + i] = norm;
        let value = z / sqrt(norm);
        scratch[p(8u) + 2u * i] = value.x;
        scratch[p(8u) + 2u * i + 1u] = value.y;
    }
}

fn histories(i: u32) {
    if i < 96u {
        var value: f32;
        if i < 64u { value = state_in[1088u + i + 32u]; }
        else { value = scratch[p(7u) + i - 64u]; }
        scratch[p(9u) + i] = value;
        state_out[1088u + i] = value;
    }
    if i < 576u {
        let channel = i / 288u;
        let position = i % 288u;
        let time = position / 96u;
        let f = position % 96u;
        var value: f32;
        if time < 2u { value = state_in[1184u + channel * 288u + (time + 1u) * 96u + f]; }
        else { value = scratch[p(8u) + 2u * f + channel]; }
        scratch[p(10u) + i] = value;
        state_out[1184u + i] = value;
    }
    if i < 4810u {
        if i < 3848u { state_out[32480u + i] = state_in[32480u + i + 962u]; }
        else { state_out[32480u + i] = scratch[p(6u) + i - 3848u]; }
    }
    if i < 6734u {
        if i < 5772u { state_out[37290u + i] = state_in[37290u + i + 962u]; }
        else { state_out[37290u + i] = scratch[p(6u) + i - 5772u]; }
    }
}

fn filter_spectrum(f: u32) {
    let band = u32(weights[p(21u) + f]);
    var wet = vec2<f32>(state_out[37290u + 4u * 962u + 2u * f],
                        state_out[37290u + 4u * 962u + 2u * f + 1u]);
    wet *= scratch[p(13u) + band];
    if f < 96u {
        wet = vec2<f32>(0.0);
        for (var tap = 0u; tap < 5u; tap += 1u) {
            let s = 32480u + tap * 962u + 2u * f;
            let c = p(14u) + (tap * 96u + f) * 2u;
            let z = vec2<f32>(state_out[s], state_out[s + 1u]);
            let coefficient = vec2<f32>(scratch[c], scratch[c + 1u]);
            wet += vec2<f32>(z.x * coefficient.x - z.y * coefficient.y,
                             z.x * coefficient.y + z.y * coefficient.x);
        }
    }
    // Disabled attenuation is the wet signal, not a dry/bypass special case.
    if p(23u) != 0u {
        let dry = vec2<f32>(state_in[32480u + 2u * 962u + 2u * f],
                            state_in[32480u + 2u * 962u + 2u * f + 1u]);
        let alpha = bitcast<f32>(p(22u));
        let difference = dry - wet;
        if alpha < 0.5 { wet = wet + alpha * difference; }
        else { wet = dry - (1.0 - alpha) * difference; }
    }
    scratch[p(15u) + 2u * f] = wet.x;
    scratch[p(15u) + 2u * f + 1u] = wet.y;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    // All 128 lanes must reach the feature initialization/reduction barriers.
    if p(0u) == 1u { features(local_id.x); return; }
    let i = global_id.x;
    if i >= p(1u) { return; }
    if p(0u) == 0u {
        var value: f32;
        if i < 480u { value = state_in[128u + i]; }
        else { value = scratch[p(2u) + i - 480u]; }
        scratch[p(5u) + i] = value * weights[p(4u) + i];
        if i < 480u { state_out[128u + i] = scratch[p(2u) + i]; }
    } else if p(0u) == 2u { histories(i); }
    else if p(0u) == 3u {
        let channel = i / 480u;
        let time = (i / 96u) % 5u;
        let f = i % 96u;
        var value: f32;
        if time < 4u { value = state_in[1760u + i + 96u]; }
        else { value = scratch[p(12u) + channel * 96u + f]; }
        state_out[1760u + i] = value;
        scratch[p(11u) + i] = value;
    } else if p(0u) == 4u { filter_spectrum(i); }
    else if p(0u) == 5u {
        scratch[p(3u) + i] = scratch[p(16u) + i] * weights[p(4u) + i] + state_in[608u + i];
        state_out[608u + i] = scratch[p(16u) + i + 480u] * weights[p(4u) + i + 480u];
    }
}
