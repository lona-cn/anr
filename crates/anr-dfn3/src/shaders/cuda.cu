// Shared CUDA/NVRTC source. Five-pointer ABI; launch sizes network=64, dsp=128,
// fft=128. Compile FP32 with --fmad=false, without --use_fast_math.
static __device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
static __device__ __forceinline__ float activate(float x, unsigned kind) {
    if (kind == 1) return fmaxf(x, 0.0f);
    if (kind == 2) return sigmoid(x);
    if (kind == 3) return tanhf(x);
    return x;
}

extern "C" __global__ void network(const float* weights, const float* state_in,
    float* state_out, float* scratch, const unsigned* p) {
    __shared__ float gru_input[256];
    __shared__ float gru_old_state[256];
    unsigned index = blockIdx.x * 64u + threadIdx.x;
    if (p[0] == 5) {
        // All lanes load the complete shared vectors before the bounds check.
        for (unsigned round = 0; round < 4; ++round) {
            unsigned k = threadIdx.x + round * 64;
            gru_input[k] = scratch[p[2] + k];
            gru_old_state[k] = state_in[p[8] + k];
        }
        __syncthreads();
        if (index >= p[1]) return;
        float wz = weights[p[10] + index];
        float wr = weights[p[10] + 256 + index];
        float wh = weights[p[10] + 512 + index];
        float rz = weights[p[10] + 768 + index];
        float rr = weights[p[10] + 1024 + index];
        float rh = weights[p[10] + 1280 + index];
        for (unsigned k = 0; k < 256; ++k) {
            unsigned base = p[4] + k * 1536 + index;
            float x = gru_input[k], old = gru_old_state[k];
            wz += x * weights[base];
            wr += x * weights[base + 256];
            wh += x * weights[base + 512];
            rz += old * weights[base + 768];
            rr += old * weights[base + 1024];
            rh += old * weights[base + 1280];
        }
        float z = sigmoid(wz + rz), r = sigmoid(wr + rr);
        float candidate = tanhf(wh + r * rh);
        float value = (1.0f - z) * candidate + z * gru_old_state[index];
        scratch[p[3] + index] = value;
        state_out[p[8] + index] = value;
        return;
    }
    if (index >= p[1]) return;
    if (p[0] <= 1) {
        bool transposed = p[0] == 1;
        unsigned ow = index % p[11], oh = (index / p[11]) % p[10];
        unsigned oc = index / (p[11] * p[10]);
        unsigned inputs_per_group = p[6] / p[18];
        unsigned outputs_per_group = p[9] / p[18];
        unsigned group = oc / outputs_per_group;
        float sum = p[5] == 0xffffffffu ? 0.0f : weights[p[5] + oc];
        for (unsigned ic = 0; ic < inputs_per_group; ++ic) {
            unsigned input_channel = group * inputs_per_group + ic;
            for (unsigned kh = 0; kh < p[12]; ++kh) {
                int iy;
                if (transposed) {
                    int t = int(oh) + int(p[16]) - int(kh);
                    if (t < 0 || t % int(p[14]) != 0) continue;
                    iy = t / int(p[14]);
                } else iy = int(oh * p[14] + kh) - int(p[16]);
                if (iy < 0 || iy >= int(p[7])) continue;
                for (unsigned kw = 0; kw < p[13]; ++kw) {
                    int ix;
                    if (transposed) {
                        int t = int(ow) + int(p[17]) - int(kw);
                        if (t < 0 || t % int(p[15]) != 0) continue;
                        ix = t / int(p[15]);
                    } else ix = int(ow * p[15] + kw) - int(p[17]);
                    if (ix < 0 || ix >= int(p[8])) continue;
                    unsigned wi;
                    if (transposed) wi = ((input_channel * outputs_per_group + oc % outputs_per_group) * p[12] + kh) * p[13] + kw;
                    else wi = ((oc * inputs_per_group + ic) * p[12] + kh) * p[13] + kw;
                    unsigned si = (input_channel * p[7] + unsigned(iy)) * p[8] + unsigned(ix);
                    sum += scratch[p[2] + si] * weights[p[4] + wi];
                }
            }
        }
        scratch[p[3] + index] = activate(sum, p[19]);
    } else if (p[0] == 2) {
        unsigned group = index / p[8], output = index % p[8];
        float sum = 0.0f;
        for (unsigned i = 0; i < p[7]; ++i) {
            unsigned wi = (group * p[7] + i) * p[8] + output;
            sum += scratch[p[2] + group * p[7] + i] * weights[p[4] + wi];
        }
        scratch[p[3] + index] = activate(sum, p[19]);
    } else if (p[0] == 3) {
        scratch[p[3] + index] = scratch[p[2] + index] + scratch[p[4] + index];
    } else if (p[0] == 4) {
        unsigned source;
        if (p[8] == 0) source = (index % p[6]) * p[7] + index / p[6];
        else source = (index % p[7]) * p[6] + index / p[7];
        scratch[p[3] + index] = scratch[p[2] + source];
    } else if (p[0] == 6) {
        unsigned component = index % 2, frequency = (index / 2) % 96;
        unsigned channel = (index / 192) * 2 + component;
        scratch[p[3] + index] = scratch[p[2] + frequency * 10 + channel]
            + scratch[p[4] + channel * 96 + frequency];
    }
}

extern "C" __global__ void dsp(const float* weights, const float* state_in,
    float* state_out, float* scratch, const unsigned* p) {
    __shared__ unsigned nonzero_mean;
    __shared__ unsigned nonzero_norm;
    unsigned i = blockIdx.x * 128u + threadIdx.x;
    if (p[0] == 1) {
        i = threadIdx.x;
        if (i == 0) { nonzero_mean = 0; nonzero_norm = 0; }
        __syncthreads();
        if (i < 32 && state_in[i] != 0.0f) atomicOr(&nonzero_mean, 1u);
        if (i < 96 && state_in[32 + i] != 0.0f) atomicOr(&nonzero_norm, 1u);
        __syncthreads();
        const float beta = 0.009999990463256836f;
        if (i < 32) {
            unsigned start = unsigned(weights[p[19] + i]);
            unsigned end = unsigned(weights[p[19] + 32 + i]);
            float power = 0.0f;
            for (unsigned f = start; f < end; ++f) {
                float x = scratch[p[6] + 2 * f], y = scratch[p[6] + 2 * f + 1];
                power += (x * x + y * y) * weights[p[20] + f];
            }
            float energy = 10.0f * (logf(power + 1.000000013351432e-10f) / 2.3025851249694824f);
            float old = nonzero_mean == 0 ? weights[p[17] + i] : state_in[i];
            float mean = old - (old - energy) * beta;
            state_out[i] = mean;
            scratch[p[7] + i] = (energy - mean) / 40.0f;
        }
        if (i < 96) {
            float x = scratch[p[6] + 2 * i], y = scratch[p[6] + 2 * i + 1];
            float magnitude = sqrtf(x * x + y * y);
            float old = nonzero_norm == 0 ? weights[p[18] + i] : state_in[32 + i];
            float norm = old - (old - magnitude) * beta;
            state_out[32 + i] = norm;
            float denominator = sqrtf(norm);
            scratch[p[8] + 2 * i] = x / denominator;
            scratch[p[8] + 2 * i + 1] = y / denominator;
        }
        return;
    }
    if (i >= p[1]) return;
    if (p[0] == 0) {
        float value = i < 480 ? state_in[128 + i] : scratch[p[2] + i - 480];
        scratch[p[5] + i] = value * weights[p[4] + i];
        if (i < 480) state_out[128 + i] = scratch[p[2] + i];
    } else if (p[0] == 2) {
        if (i < 96) {
            float value = i < 64 ? state_in[1088 + i + 32] : scratch[p[7] + i - 64];
            scratch[p[9] + i] = value;
            state_out[1088 + i] = value;
        }
        if (i < 576) {
            unsigned channel = i / 288, position = i % 288;
            unsigned time = position / 96, f = position % 96;
            float value = time < 2 ? state_in[1184 + channel * 288 + (time + 1) * 96 + f]
                : scratch[p[8] + 2 * f + channel];
            scratch[p[10] + i] = value;
            state_out[1184 + i] = value;
        }
        if (i < 4810) state_out[32480 + i] = i < 3848 ? state_in[32480 + i + 962] : scratch[p[6] + i - 3848];
        if (i < 6734) state_out[37290 + i] = i < 5772 ? state_in[37290 + i + 962] : scratch[p[6] + i - 5772];
    } else if (p[0] == 3) {
        unsigned channel = i / 480, time = (i / 96) % 5, f = i % 96;
        float value = time < 4 ? state_in[1760 + i + 96] : scratch[p[12] + channel * 96 + f];
        state_out[1760 + i] = value;
        scratch[p[11] + i] = value;
    } else if (p[0] == 4) {
        unsigned band = unsigned(weights[p[21] + i]);
        float gain = scratch[p[13] + band];
        float x = state_out[37290 + 4 * 962 + 2 * i] * gain;
        float y = state_out[37290 + 4 * 962 + 2 * i + 1] * gain;
        if (i < 96) {
            x = 0.0f; y = 0.0f;
            for (unsigned tap = 0; tap < 5; ++tap) {
                unsigned s = 32480 + tap * 962 + 2 * i;
                unsigned c = p[14] + (tap * 96 + i) * 2;
                float zx = state_out[s], zy = state_out[s + 1];
                float cx = scratch[c], cy = scratch[c + 1];
                x += zx * cx - zy * cy;
                y += zx * cy + zy * cx;
            }
        }
        if (p[23] != 0) {
            float dx = state_in[32480 + 2 * 962 + 2 * i], dy = state_in[32480 + 2 * 962 + 2 * i + 1];
            float alpha = __uint_as_float(p[22]);
            float diff_x = dx - x, diff_y = dy - y;
            if (alpha < 0.5f) { x = x + alpha * diff_x; y = y + alpha * diff_y; }
            else { x = dx - (1.0f - alpha) * diff_x; y = dy - (1.0f - alpha) * diff_y; }
        }
        scratch[p[15] + 2 * i] = x;
        scratch[p[15] + 2 * i + 1] = y;
    } else if (p[0] == 5) {
        scratch[p[3] + i] = scratch[p[16] + i] * weights[p[4] + i] + state_in[608 + i];
        state_out[608 + i] = scratch[p[16] + i + 480] * weights[p[4] + i + 480];
    }
}

extern "C" __global__ void fft(const float* weights, const float* state_in,
    float* state_out, float* scratch, const unsigned* p) {
    __shared__ float2 a[960];
    __shared__ float2 b[960];
    const unsigned radix[5] = {5, 3, 4, 4, 4};
    unsigned lane = threadIdx.x;
    for (unsigned i = lane; i < 960; i += 128) {
        unsigned index = i, stride = 960, source = 0;
        for (unsigned stage = 0; stage < 5; ++stage) {
            stride /= radix[stage];
            source += (index % radix[stage]) * stride;
            index /= radix[stage];
        }
        if (p[0] == 0) a[i] = make_float2(scratch[p[2] + source], 0.0f);
        else {
            unsigned frequency = source > 480 ? 960 - source : source;
            float2 value = make_float2(scratch[p[2] + 2 * frequency], scratch[p[2] + 2 * frequency + 1]);
            if (frequency == 0 || frequency == 480) value.y = 0.0f;
            if (source > 480) value.y = -value.y;
            a[i] = value;
        }
    }
    __syncthreads();
    unsigned previous = 1;
    for (unsigned stage = 0; stage < 5; ++stage) {
        unsigned r = radix[stage], extent = previous * r;
        for (unsigned i = lane; i < 960; i += 128) {
            unsigned base = (i / extent) * extent, k = i % previous;
            unsigned digit = (i % extent) / previous;
            float2 sum = make_float2(0.0f, 0.0f);
            for (unsigned q = 0; q < r; ++q) {
                unsigned phase = (q * k * (960 / extent) + q * digit * (960 / r)) % 960;
                float2 twiddle = make_float2(weights[p[4] + 2 * phase], weights[p[4] + 2 * phase + 1]);
                if (p[0] != 0) twiddle.y = -twiddle.y;
                unsigned source = base + k + q * previous;
                float2 value = stage % 2 == 0 ? a[source] : b[source];
                sum.x += value.x * twiddle.x - value.y * twiddle.y;
                sum.y += value.x * twiddle.y + value.y * twiddle.x;
            }
            if (stage % 2 == 0) b[i] = sum;
            else a[i] = sum;
        }
        __syncthreads();
        previous = extent;
    }
    if (p[0] == 0) {
        for (unsigned i = lane; i < 481; i += 128) {
            scratch[p[3] + 2 * i] = b[i].x * (1.0f / 960.0f);
            scratch[p[3] + 2 * i + 1] = b[i].y * (1.0f / 960.0f);
        }
    } else {
        for (unsigned i = lane; i < 960; i += 128) scratch[p[3] + i] = b[i].x;
    }
}
