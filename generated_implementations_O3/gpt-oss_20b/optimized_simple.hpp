#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <arm_neon.h>

// Very simple version: process 8 columns per iteration, update shift and word
// index manually with no table look‑ups.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    const size_t block  = 8;
    std::memset(C, 0, M * K * sizeof(float));
    for (size_t i = 0; i < M; ++i) {
        float* c_row = &C[i * K];
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const uint32_t* b_row = &B[p * K_ints];
            size_t b_idx = 0;
            unsigned shift = 0;
            uint32_t packed = b_row[b_idx];
            for (size_t j = 0; j + block <= K; j += block) {
                unsigned bits = (packed >> shift) & 0xFFu;
                unsigned s[8];
                for (int t = 0; t < 8; ++t) s[t] = (bits >> t) & 1u;
                float32x4_t c_low = vld1q_f32(&c_row[j]);
                float32x4_t c_high = vld1q_f32(&c_row[j + 4]);
                float32x4_t a_vec = vdupq_n_f32(a_val);
                float32x4_t s_low = {s[0]?1.0f:-1.0f, s[1]?1.0f:-1.0f, s[2]?1.0f:-1.0f, s[3]?1.0f:-1.0f};
                float32x4_t s_high = {s[4]?1.0f:-1.0f, s[5]?1.0f:-1.0f, s[6]?1.0f:-1.0f, s[7]?1.0f:-1.0f};
                c_low = vaddq_f32(c_low, vmulq_f32(a_vec, s_low));
                c_high = vaddq_f32(c_high, vmulq_f32(a_vec, s_high));
                vst1q_f32(&c_row[j], c_low);
                vst1q_f32(&c_row[j+4], c_high);
                shift += 8;
                if (shift >= 32){ shift -= 32; ++b_idx; packed = b_row[b_idx]; }
            }
        }
    }
}
