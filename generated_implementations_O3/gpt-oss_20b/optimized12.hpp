#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <arm_neon.h>

// Optimized matrix multiplication using NEON and per‑block sign generation.
// No global data: all lookup is done locally for each 8‑column block.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;   // words per row of B
    const size_t block  = 8;        // columns processed per SIMD step

    std::memset(C, 0, M * K * sizeof(float)); // clear C

    for (size_t i = 0; i < M; ++i) {
        float* c_row = &C[i * K];
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const uint32_t* b_row = &B[p * K_ints];
            size_t b_idx = 0;      // current word index
            unsigned shift = 0;     // bit offset within current word

            size_t j = 0;
            for (; j + block <= K; j += block) {
                // Load current 32‑bit word and, if needed, the next one
                uint32_t w0 = b_row[b_idx];
                uint32_t w1 = b_row[b_idx + 1];
                uint64_t combined = (uint64_t)w0 | ((uint64_t)w1 << 32);
                uint16_t mask = (uint16_t)((combined >> shift) & 0xFFFFU);

                // Build 8‑float sign array from mask
                float signs[8];
                for (int t = 0; t < 8; ++t) {
                    signs[t] = ((mask >> t) & 1U) ? 1.0f : -1.0f;
                }

                // Load C block
                float32x4_t c_low  = vld1q_f32(&c_row[j]);
                float32x4_t c_high = vld1q_f32(&c_row[j + 4]);

                float32x4_t a_vec = vdupq_n_f32(a_val);

                // Load sign vectors
                float32x4_t s_low  = vld1q_f32(&signs[0]);
                float32x4_t s_high = vld1q_f32(&signs[4]);

                // Accumulate
                c_low  = vaddq_f32(c_low,  vmulq_f32(a_vec, s_low));
                c_high = vaddq_f32(c_high, vmulq_f32(a_vec, s_high));

                vst1q_f32(&c_row[j], c_low);
                vst1q_f32(&c_row[j + 4], c_high);

                // Advance shift and word index
                shift += block;
                while (shift >= 32) {
                    shift -= 32;
                    ++b_idx;
                }
            }

            // Tail columns <8
            for (; j < K; ++j) {
                uint32_t w = b_row[b_idx];
                unsigned bit = (w >> shift) & 1U;
                float sign = bit ? 1.0f : -1.0f;
                c_row[j] += a_val * sign;

                ++shift;
                if (shift == 32) {
                    shift = 0;
                    ++b_idx;
                }
            }
        }
    }
}
