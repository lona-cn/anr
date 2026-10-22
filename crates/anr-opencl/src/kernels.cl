// Direct FP32 port of anr-dfn3's fixed WGSL programs. The host instantiates
// this template for each Model dispatch; only attenuation words stay dynamic.
#ifndef DFN3_COMMON
#define DFN3_COMMON
#pragma OPENCL FP_CONTRACT OFF
#define ARGS __global const float *weights, __global const float *state_in, __global float *state_out, __global float *scratch, __global const uint *parameters
#define p(i) DFN3_P_ ## i

inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float activate(float x, uint kind) {
    if (kind == 1u) return fmax(x, 0.0f);
    if (kind == 2u) return sigmoid(x);
    if (kind == 3u) return tanh(x);
    return x;
}
inline float2 product(float2 x, float2 y) {
    return (float2)(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
}
#endif

#if DFN3_SHADER == 0

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void DFN3_ENTRY(ARGS) {
    uint index = (uint)get_global_id(0);
#if p(0u) == 5u
    __local float gru_input[256];
    __local float gru_old_state[256];
    uint lane = (uint)get_local_id(0);
    // Barrier must precede every bounds return, even in a partial workgroup.
    {
        for (uint round = 0u; round < 4u; ++round) {
            uint k = lane + round * 64u;
            gru_input[k] = scratch[p(2u) + k];
            gru_old_state[k] = state_in[p(8u) + k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (index >= p(1u)) return;
        float wz = weights[p(10u) + index];
        float wr = weights[p(10u) + 256u + index];
        float wh = weights[p(10u) + 512u + index];
        float rz = weights[p(10u) + 768u + index];
        float rr = weights[p(10u) + 1024u + index];
        float rh = weights[p(10u) + 1280u + index];
        for (uint k = 0u; k < 256u; ++k) {
            uint base = p(4u) + k * 1536u + index;
            float x = gru_input[k];
            float old = gru_old_state[k];
            wz += x * weights[base];
            wr += x * weights[base + 256u];
            wh += x * weights[base + 512u];
            rz += old * weights[base + 768u];
            rr += old * weights[base + 1024u];
            rh += old * weights[base + 1280u];
        }
        // ONNX linear_before_reset=1: reset multiplies the full recurrent term.
        float z = sigmoid(wz + rz);
        float r = sigmoid(wr + rr);
        float candidate = tanh(wh + r * rh);
        float value = (1.0f - z) * candidate + z * gru_old_state[index];
        scratch[p(3u) + index] = value;
        state_out[p(8u) + index] = value;
        return;
    }
#else
    if (index >= p(1u)) return;
#if p(0u) == 0u || p(0u) == 1u
    {
        uint ow = index % p(11u);
        uint oh = (index / p(11u)) % p(10u);
        uint oc = index / (p(11u) * p(10u));
        uint inputs_per_group = p(6u) / p(18u);
        uint outputs_per_group = p(9u) / p(18u);
        uint group = oc / outputs_per_group;
        float sum = 0.0f;
        if (p(5u) != 0xffffffffu) sum = weights[p(5u) + oc];
        for (uint ic = 0u; ic < inputs_per_group; ++ic) {
            uint input_channel = group * inputs_per_group + ic;
            for (uint kh = 0u; kh < p(12u); ++kh) {
                int iy;
#if p(0u) == 1u
                int t = (int)oh + (int)p(16u) - (int)kh;
                if (t < 0 || t % (int)p(14u) != 0) continue;
                iy = t / (int)p(14u);
#else
                iy = (int)(oh * p(14u) + kh) - (int)p(16u);
#endif
                if (iy < 0 || iy >= (int)p(7u)) continue;
                for (uint kw = 0u; kw < p(13u); ++kw) {
                    int ix;
#if p(0u) == 1u
                    int t = (int)ow + (int)p(17u) - (int)kw;
                    if (t < 0 || t % (int)p(15u) != 0) continue;
                    ix = t / (int)p(15u);
#else
                    ix = (int)(ow * p(15u) + kw) - (int)p(17u);
#endif
                    if (ix < 0 || ix >= (int)p(8u)) continue;
                    uint wi;
#if p(0u) == 1u
                    wi = ((input_channel * outputs_per_group + oc % outputs_per_group)
                        * p(12u) + kh) * p(13u) + kw;
#else
                    wi = ((oc * inputs_per_group + ic) * p(12u) + kh) * p(13u) + kw;
#endif
                    uint si = (input_channel * p(7u) + (uint)iy) * p(8u) + (uint)ix;
                    sum += scratch[p(2u) + si] * weights[p(4u) + wi];
                }
            }
        }
        scratch[p(3u) + index] = activate(sum, p(19u));
    }
#elif p(0u) == 2u
    {
        uint group = index / p(8u);
        uint output_in_group = index % p(8u);
        float sum = 0.0f;
        for (uint i = 0u; i < p(7u); ++i) {
            uint wi = (group * p(7u) + i) * p(8u) + output_in_group;
            sum += scratch[p(2u) + group * p(7u) + i] * weights[p(4u) + wi];
        }
        scratch[p(3u) + index] = activate(sum, p(19u));
    }
#elif p(0u) == 3u
    {
        scratch[p(3u) + index] = scratch[p(2u) + index] + scratch[p(4u) + index];
    }
#elif p(0u) == 4u
    {
        uint source;
#if p(8u) == 0u
        uint channel = index % p(6u);
        uint spatial = index / p(6u);
        source = channel * p(7u) + spatial;
#else
        uint channel = index / p(7u);
        uint spatial = index % p(7u);
        source = spatial * p(6u) + channel;
#endif
        scratch[p(3u) + index] = scratch[p(2u) + source];
    }
#elif p(0u) == 6u
    {
        uint component = index % 2u;
        uint frequency = (index / 2u) % 96u;
        uint tap = index / 192u;
        uint channel = tap * 2u + component;
        scratch[p(3u) + index] = scratch[p(2u) + frequency * 10u + channel]
            + scratch[p(4u) + channel * 96u + frequency];
    }
#endif
#endif
}
#endif

#if DFN3_SHADER == 1

#define spectrum(f) ((float2)(scratch[p(6u) + 2u * (f)], scratch[p(6u) + 2u * (f) + 1u]))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void DFN3_ENTRY(ARGS) {
    __local uint nonzero_mean;
    __local uint nonzero_norm;
    uint i = (uint)get_global_id(0);
    uint mode = p(0u);
    if (mode == 1u) {
        i = (uint)get_local_id(0);
        if (i == 0u) { nonzero_mean = 0u; nonzero_norm = 0u; }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (i < 32u && state_in[i] != 0.0f) atomic_or(&nonzero_mean, 1u);
        if (i < 96u && state_in[32u + i] != 0.0f) atomic_or(&nonzero_norm, 1u);
        barrier(CLK_LOCAL_MEM_FENCE);
        const float beta = 0.009999990463256836f;
        if (i < 32u) {
            uint start = (uint)weights[p(19u) + i];
            uint end = (uint)weights[p(19u) + 32u + i];
            float power = 0.0f;
            for (uint f = start; f < end; ++f) {
                float2 z = spectrum(f);
                power += (z.x * z.x + z.y * z.y) * weights[p(20u) + f];
            }
            float energy = 10.0f * (log(power + 1.000000013351432e-10f) / 2.3025851249694824f);
            float old = state_in[i];
            if (nonzero_mean == 0u) old = weights[p(17u) + i];
            float mean = old - (old - energy) * beta;
            state_out[i] = mean;
            scratch[p(7u) + i] = (energy - mean) / 40.0f;
        }
        if (i < 96u) {
            float2 z = spectrum(i);
            float magnitude = sqrt(z.x * z.x + z.y * z.y);
            float old = state_in[32u + i];
            if (nonzero_norm == 0u) old = weights[p(18u) + i];
            float norm = old - (old - magnitude) * beta;
            state_out[32u + i] = norm;
            float2 value = z / sqrt(norm);
            scratch[p(8u) + 2u * i] = value.x;
            scratch[p(8u) + 2u * i + 1u] = value.y;
        }
        return;
    }
    if (i >= p(1u)) return;
    if (mode == 0u) {
        float value;
        if (i < 480u) value = state_in[128u + i];
        else value = scratch[p(2u) + i - 480u];
        scratch[p(5u) + i] = value * weights[p(4u) + i];
        if (i < 480u) state_out[128u + i] = scratch[p(2u) + i];
    } else if (mode == 2u) {
        if (i < 96u) {
            float value;
            if (i < 64u) value = state_in[1088u + i + 32u];
            else value = scratch[p(7u) + i - 64u];
            scratch[p(9u) + i] = value;
            state_out[1088u + i] = value;
        }
        if (i < 576u) {
            uint channel = i / 288u;
            uint position = i % 288u;
            uint time = position / 96u;
            uint f = position % 96u;
            float value;
            if (time < 2u) value = state_in[1184u + channel * 288u + (time + 1u) * 96u + f];
            else value = scratch[p(8u) + 2u * f + channel];
            scratch[p(10u) + i] = value;
            state_out[1184u + i] = value;
        }
        if (i < 4810u) {
            if (i < 3848u) state_out[32480u + i] = state_in[32480u + i + 962u];
            else state_out[32480u + i] = scratch[p(6u) + i - 3848u];
        }
        if (i < 6734u) {
            if (i < 5772u) state_out[37290u + i] = state_in[37290u + i + 962u];
            else state_out[37290u + i] = scratch[p(6u) + i - 5772u];
        }
    } else if (mode == 3u) {
        uint channel = i / 480u;
        uint time = (i / 96u) % 5u;
        uint f = i % 96u;
        float value;
        if (time < 4u) value = state_in[1760u + i + 96u];
        else value = scratch[p(12u) + channel * 96u + f];
        state_out[1760u + i] = value;
        scratch[p(11u) + i] = value;
    } else if (mode == 4u) {
        uint f = i;
        uint band = (uint)weights[p(21u) + f];
        float2 wet = (float2)(state_out[37290u + 4u * 962u + 2u * f],
                              state_out[37290u + 4u * 962u + 2u * f + 1u]);
        wet *= scratch[p(13u) + band];
        if (f < 96u) {
            wet = (float2)(0.0f);
            for (uint tap = 0u; tap < 5u; ++tap) {
                uint s = 32480u + tap * 962u + 2u * f;
                uint c = p(14u) + (tap * 96u + f) * 2u;
                float2 z = (float2)(state_out[s], state_out[s + 1u]);
                float2 coefficient = (float2)(scratch[c], scratch[c + 1u]);
                wet += (float2)(z.x * coefficient.x - z.y * coefficient.y,
                                 z.x * coefficient.y + z.y * coefficient.x);
            }
        }
        if (p(23u) != 0u) {
            float2 dry = (float2)(state_in[32480u + 2u * 962u + 2u * f],
                                  state_in[32480u + 2u * 962u + 2u * f + 1u]);
            float alpha = as_float(p(22u));
            float2 difference = dry - wet;
            if (alpha < 0.5f) wet = wet + alpha * difference;
            else wet = dry - (1.0f - alpha) * difference;
        }
        scratch[p(15u) + 2u * f] = wet.x;
        scratch[p(15u) + 2u * f + 1u] = wet.y;
    } else if (mode == 5u) {
        scratch[p(3u) + i] = scratch[p(16u) + i] * weights[p(4u) + i] + state_in[608u + i];
        state_out[608u + i] = scratch[p(16u) + i + 480u] * weights[p(4u) + i + 480u];
    }
}

#undef spectrum
#endif

#if DFN3_SHADER == 2
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void DFN3_ENTRY(ARGS) {
    __local float2 a[960];
    __local float2 b[960];
    const uint radix[5] = {5u, 3u, 4u, 4u, 4u};
    uint lane = (uint)get_local_id(0);
    for (uint i = lane; i < 960u; i += 128u) {
        uint index = i;
        uint stride = 960u;
        uint source = 0u;
        for (uint stage = 0u; stage < 5u; ++stage) {
            stride /= radix[stage];
            source += (index % radix[stage]) * stride;
            index /= radix[stage];
        }
        if (p(0u) == 0u) a[i] = (float2)(scratch[p(2u) + source], 0.0f);
        else {
            uint frequency = source;
            if (source > 480u) frequency = 960u - source;
            float2 value = (float2)(scratch[p(2u) + 2u * frequency],
                                    scratch[p(2u) + 2u * frequency + 1u]);
            if (frequency == 0u || frequency == 480u) value.y = 0.0f;
            if (source > 480u) value.y = -value.y;
            a[i] = value;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint previous = 1u;
    for (uint stage = 0u; stage < 5u; ++stage) {
        uint r = radix[stage];
        uint extent = previous * r;
        for (uint i = lane; i < 960u; i += 128u) {
            uint base = (i / extent) * extent;
            uint k = i % previous;
            uint output_digit = (i % extent) / previous;
            float2 sum = (float2)(0.0f);
            for (uint q = 0u; q < r; ++q) {
                uint phase = (q * k * (960u / extent) + q * output_digit * (960u / r)) % 960u;
                float2 twiddle = (float2)(weights[p(4u) + 2u * phase], weights[p(4u) + 2u * phase + 1u]);
                if (p(0u) != 0u) twiddle.y = -twiddle.y;
                uint source = base + k + q * previous;
                float2 value;
                if (stage % 2u == 0u) value = a[source];
                else value = b[source];
                sum += product(value, twiddle);
            }
            if (stage % 2u == 0u) b[i] = sum;
            else a[i] = sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        previous = extent;
    }
    if (p(0u) == 0u) {
        for (uint i = lane; i < 481u; i += 128u) {
            float2 value = b[i] * (1.0f / 960.0f);
            scratch[p(3u) + 2u * i] = value.x;
            scratch[p(3u) + 2u * i + 1u] = value.y;
        }
    } else {
        // Unscaled inverse cancels the original graph's post-IRFFT *960.
        for (uint i = lane; i < 960u; i += 128u) scratch[p(3u) + i] = b[i].x;
    }
}
#endif
