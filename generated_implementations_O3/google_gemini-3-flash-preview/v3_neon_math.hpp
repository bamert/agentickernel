#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Use a small local buffer for precomputed 2*a values to help compiler
    // But directly processing is fine with NEON.
    float32x4_t zero = vdupq_n_f32(0.0f);

    for (size_t i = 0; i < M; ++i) {
        float* Ci = &C[i * K];
        const float* Ai = &A[i * K];

        // Step 1: Pre-calculate Row sum for the "bit ? 1 : -1" trick
        // sum = sum(a * (2*bit - 1)) = 2 * sum(a * bit) - sum(a)
        float row_sum_A = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            row_sum_A += Ai[p];
        }
        
        // Initialize Row Ci with -row_sum_A
        float32x4_t neg_sum_vec = vdupq_n_f32(-row_sum_A);
        for (size_t j = 0; j < K; j += 4) {
            vst1q_f32(&Ci[j], neg_sum_vec);
        }

        // Step 2: Accumulate 2 * Ai[p] where bit is 1
        for (size_t p = 0; p < K; ++p) {
            float two_a_val = Ai[p] * 2.0f;
            float32x4_t two_a = vdupq_n_f32(two_a_val);
            const uint32_t* Bp = &B[p * K_ints];

            for (size_t kj = 0; kj < K_ints; ++kj) {
                uint32_t bits = Bp[kj];
                if (bits == 0) continue;
                
                float* Cij = &Ci[kj * 32];

                // Unroll 32 bits into 8 NEON vectors
                for (int vec_idx = 0; vec_idx < 8; ++vec_idx) {
                    uint32_t b4 = (bits >> (vec_idx * 4)) & 0xF;
                    if (b4 == 0) continue;

                    float32x4_t c_vec = vld1q_f32(&Cij[vec_idx * 4]);
                    
                    // Manual extraction of 4 bits to create a mask
                    // We can use a small lookup if needed, but let's try direct first
                    uint32_t m0 = (b4 & 1) ? 0xFFFFFFFF : 0;
                    uint32_t m1 = (b4 & 2) ? 0xFFFFFFFF : 0;
                    uint32_t m2 = (b4 & 4) ? 0xFFFFFFFF : 0;
                    uint32_t m3 = (b4 & 8) ? 0xFFFFFFFF : 0;

                    uint32_t mask_arr[4] = {m0, m1, m2, m3};
                    uint32x4_t maskv = vld1q_u32(mask_arr);
                    
                    float32x4_t masked_two_a = vreinterpretq_f32_u32(vandq_u32(maskv, vreinterpretq_u32_f32(two_a)));
                    c_vec = vaddq_f32(c_vec, masked_two_a);
                    vst1q_f32(&Cij[vec_idx * 4], c_vec);
                }
            }
        }
    }
}
