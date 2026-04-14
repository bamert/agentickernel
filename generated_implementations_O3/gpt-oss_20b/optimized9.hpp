#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <arm_neon.h>

// Further optimisation of optimized4.hpp: add prefetching of B rows and
// current C block to reduce memory stalls. The core algorithm remains the
// same – 8 columns processed per SIMD iteration.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    const size_t block   = 8; // processed per SIMD step

    // Zero output matrix C once. Using memset is fast for large buffers.
    std::memset(C, 0, M * K * sizeof(float));

    for (size_t i = 0; i < M; ++i) {
        float* c_row = &C[i * K];
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const uint32_t* b_row = &B[p * K_ints];
            size_t b_index = 0;
            uint32_t packed = b_row[b_index];
            unsigned shift  = 0;

            // Prefetch the next B row to hide its latency.
            if (p + 1 < K) {
                __builtin_prefetch(&B[(p + 1) * K_ints], 0, 3);
            }

            size_t j = 0;
            for (; j + block <= K; j += block) {
                // Prefetch the next C block – we are about to write here.
                __builtin_prefetch(&c_row[j + block], 0, 1);

                // Load the current C block.
                float32x4_t c_low  = vld1q_f32(&c_row[j]);
                float32x4_t c_high = vld1q_f32(&c_row[j + 4]);

                float32x4_t a_vec = vdupq_n_f32(a_val);

                // Extract mask for the next 8 bits.
                unsigned mask = (packed >> shift) & 0xFFU;

                // Prepare sign vectors.
                float32x4_t s_low  = vdupq_n_f32((mask       ) & 1 ? 1.0f : -1.0f);
                float32x4_t s_mid  = vdupq_n_f32((mask >> 1) & 1 ? 1.0f : -1.0f);
                float32x4_t s_mid2 = vdupq_n_f32((mask >> 2) & 1 ? 1.0f : -1.0f);
                float32x4_t s_mid3 = vdupq_n_f32((mask >> 3) & 1 ? 1.0f : -1.0f);
                float32x4_t s_mid4 = vdupq_n_f32((mask >> 4) & 1 ? 1.0f : -1.0f);
                float32x4_t s_mid5 = vdupq_n_f32((mask >> 5) & 1 ? 1.0f : -1.0f);
                float32x4_t s_mid6 = vdupq_n_f32((mask >> 6) & 1 ? 1.0f : -1.0f);
                float32x4_t s_mid7 = vdupq_n_f32((mask >> 7) & 1 ? 1.0f : -1.0f);

                // Accumulate the eight contributions.
                c_low  = vaddq_f32(c_low, vmulq_f32(a_vec, s_low));
                c_low  = vaddq_f32(c_low, vmulq_f32(a_vec, s_mid));
                c_low  = vaddq_f32(c_low, vmulq_f32(a_vec, s_mid2));
                c_low  = vaddq_f32(c_low, vmulq_f32(a_vec, s_mid3));
                c_high = vaddq_f32(c_high, vmulq_f32(a_vec, s_mid4));
                c_high = vaddq_f32(c_high, vmulq_f32(a_vec, s_mid5));
                c_high = vaddq_f32(c_high, vmulq_f32(a_vec, s_mid6));
                c_high = vaddq_f32(c_high, vmulq_f32(a_vec, s_mid7));

                vst1q_f32(&c_row[j], c_low);
                vst1q_f32(&c_row[j + 4], c_high);

                // Update shift and packed for next block.
                shift += 8;
                if (shift >= 32) {
                    shift -= 32;
                    ++b_index;
                    packed = b_row[b_index];
                }
            }

            // Handle tail columns.
            for (; j < K; ++j) {
                unsigned bit = (packed >> shift) & 1U;
                float sign = bit ? 1.0f : -1.0f;
                c_row[j] += a_val * sign;

                shift = (shift + 1) & 31U;
                if (shift == 0) {
                    ++b_index;
                    packed = b_row[b_index];
                }
            }
        }
    }
}
