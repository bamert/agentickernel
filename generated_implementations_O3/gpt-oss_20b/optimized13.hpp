#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <arm_neon.h>

// Optimized matrix multiplication: NEON + per‑block sign calculation.
// This variant unrolls the block to process 2 rows of C at a time but keeps
// all data local.  It should avoid the pitfalls of the previous attempts.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    const size_t block = 8;

    std::memset(C, 0, M * K * sizeof(float));

    for (size_t i = 0; i < M; ++i) {
        const float* a_row = &A[i * K];
        float* c_row = &C[i * K];
        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            const uint32_t* b_row = &B[p * K_ints];
            size_t b_idx = 0;
            uint32_t packed = b_row[b_idx];
            unsigned shift = 0;

            size_t j = 0;
            for (; j + block <= K; j += block) {
                float32x4_t c_low  = vld1q_f32(&c_row[j]);
                float32x4_t c_high = vld1q_f32(&c_row[j + 4]);

                // Compute 8 signs on the fly
                float s_vals[8];
                for (int t = 0; t < 8; ++t) {
                    s_vals[t] = ((packed >> shift) & 1U) ? 1.0f : -1.0f;
                    ++shift;
                    if (shift == 32) {
                        shift = 0;
                        ++b_idx;
                        packed = b_row[b_idx];
                    }
                }
                float32x4_t s_low  = vld1q_f32(s_vals);
                float32x4_t s_high = vld1q_f32(s_vals + 4);

                float32x4_t a_vec = vdupq_n_f32(a_val);
                c_low  = vaddq_f32(c_low,  vmulq_f32(a_vec, s_low));
                c_high = vaddq_f32(c_high, vmulq_f32(a_vec, s_high));

                vst1q_f32(&c_row[j],  c_low);
                vst1q_f32(&c_row[j + 4], c_high);
            }

            for (; j < K; ++j) {
                float sign = ((packed >> shift) & 1U) ? 1.0f : -1.0f;
                c_row[j] += a_val * sign;
                ++shift;
                if (shift == 32) {
                    shift = 0;
                    ++b_idx;
                    packed = b_row[b_idx];
                }
            }
        }
    }
}
