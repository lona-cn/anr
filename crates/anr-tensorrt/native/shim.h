#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define ANR_EXPORT __declspec(dllexport)
#else
#define ANR_EXPORT __attribute__((visibility("default")))
#endif

struct AnrDispatch {
    uint32_t shader;
    uint32_t groups;
    uint32_t p[32];
};

extern "C" {
ANR_EXPORT void* anr_trt_create(const float* weights, size_t weight_count, uint32_t scratch_len,
    uint32_t input, uint32_t output, const AnrDispatch* dispatches, size_t count,
    size_t attenuation_dispatch, const char* source, char* error, size_t capacity) noexcept;
ANR_EXPORT int anr_trt_frame(void* instance, const float* input, float* output,
    float attenuation, int reset, char* error, size_t capacity) noexcept;
ANR_EXPORT void anr_trt_destroy(void* instance) noexcept;
ANR_EXPORT const char* anr_trt_last_error() noexcept;
}
