@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> state_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> state_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;
struct Params { values: array<vec4<u32>, 8>, }
@group(0) @binding(4) var<uniform> parameters: Params;
var<workgroup> a: array<vec2<f32>, 960>;
var<workgroup> b: array<vec2<f32>, 960>;
const RADIX: array<u32, 5> = array<u32, 5>(5u, 3u, 4u, 4u, 4u);

fn p(i: u32) -> u32 { return parameters.values[i / 4u][i % 4u]; }
fn product(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
}

// Exactly one workgroup; p[0]=inverse, p[2]=source, p[3]=destination,
// p[4]=the CPU-prepared complex twiddle table. No lane returns before a barrier.
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let lane = local_id.x;
    for (var i = lane; i < 960u; i += 128u) {
        var index = i;
        var stride = 960u;
        var source = 0u;
        for (var stage = 0u; stage < 5u; stage += 1u) {
            stride /= RADIX[stage];
            source += (index % RADIX[stage]) * stride;
            index /= RADIX[stage];
        }
        if p(0u) == 0u {
            a[i] = vec2<f32>(scratch[p(2u) + source], 0.0);
        } else {
            var frequency = source;
            if source > 480u { frequency = 960u - source; }
            var value = vec2<f32>(scratch[p(2u) + 2u * frequency],
                                  scratch[p(2u) + 2u * frequency + 1u]);
            // A real inverse ignores imaginary DC and Nyquist components.
            if frequency == 0u || frequency == 480u { value.y = 0.0; }
            if source > 480u { value.y = -value.y; }
            a[i] = value;
        }
    }
    workgroupBarrier();
    var previous = 1u;
    for (var stage = 0u; stage < 5u; stage += 1u) {
        let r = RADIX[stage];
        let extent = previous * r;
        for (var i = lane; i < 960u; i += 128u) {
            let base = (i / extent) * extent;
            let k = i % previous;
            let output_digit = (i % extent) / previous;
            var sum = vec2<f32>(0.0);
            for (var q = 0u; q < r; q += 1u) {
                let phase = (q * k * (960u / extent) + q * output_digit * (960u / r)) % 960u;
                var twiddle = vec2<f32>(weights[p(4u) + 2u * phase], weights[p(4u) + 2u * phase + 1u]);
                if p(0u) != 0u { twiddle.y = -twiddle.y; }
                let source = base + k + q * previous;
                var value: vec2<f32>;
                if stage % 2u == 0u { value = a[source]; }
                else { value = b[source]; }
                sum += product(value, twiddle);
            }
            if stage % 2u == 0u { b[i] = sum; }
            else { a[i] = sum; }
        }
        workgroupBarrier();
        previous = extent;
    }
    if p(0u) == 0u {
        for (var i = lane; i < 481u; i += 128u) {
            let value = b[i] * (1.0 / 960.0);
            scratch[p(3u) + 2u * i] = value.x;
            scratch[p(3u) + 2u * i + 1u] = value.y;
        }
    } else {
        // Unscaled inverse cancels the graph's explicit post-IRFFT *960.
        for (var i = lane; i < 960u; i += 128u) { scratch[p(3u) + i] = b[i].x; }
    }
}
