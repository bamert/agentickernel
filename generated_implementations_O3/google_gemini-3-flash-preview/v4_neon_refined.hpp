#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Outer loop: Iterate over rows of A (M)
    for (size_t i = 0; i < M; ++i) {
        float* Ci = &C[i * K];
        const float* Ai = &A[i * K];

        // Step 1: Pre-calculate Row sum of A
        // sum(A_i_p * sign) = sum(A_i_p * (2*bit - 1)) = 2 * sum(A_i_p * bit) - sum(A_i_p)
        float row_sum_A = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            row_sum_A += Ai[p];
        }
        
        // Initialize Row Ci with -row_sum_A
        float32x4_t neg_sum_vec = vdupq_n_f32(-row_sum_A);
        for (size_t j = 0; j < K; j += 4) {
            vst1q_f32(&Ci[j], neg_sum_vec);
        }

        // Step 2: Accumulate 2 * Ai[p] where B_p_j bit is 1
        for (size_t p = 0; p < K; ++p) {
            float two_a_val = Ai[p] * 2.0f;
            float32x4_t v_two_a = vdupq_n_f32(two_a_val);
            const uint32_t* Bp = &B[p * K_ints];

            for (size_t kj = 0; kj < K_ints; ++kj) {
                uint32_t bits = Bp[kj];
                if (bits == 0) continue;
                
                float* Cij = &Ci[kj * 32];
                
                // Process 32 bits as 8 groups of 4 bits
                for (int g = 0; g < 8; ++g) {
                    uint32_t b4 = (bits >> (g * 4)) & 0xF;
                    if (b4 == 0) continue;

                    float32x4_t v_c = vld1q_f32(Cij + g * 4);
                    
                    // Create mask using NEON
                    // We can move 4 bits to a SIMD register and compare
                    uint32_t mask_bits[4];
                    mask_bits[0] = (b4 & 1) ? 0xFFFFFFFF : 0;
                    mask_bits[1] = (b4 & 2) ? 0xFFFFFFFF : 0;
                    mask_bits[2] = (b4 & 4) ? 0xFFFFFFFF : 0;
                    mask_bits[3] = (b4 & 8) ? 0xFFFFFFFF : 0;
                    
                    uint32x4_t v_mask = vld1q_u32(mask_bits);
                    v_c = vaddq_f32(v_c, vreinterpretq_f32_u32(vandq_u32(v_mask, vreinterpretq_u32_f32(v_two_a))));
                    
                    vst1q_f32(Cij + g * 4, v_c);
                }
            }
        }
    }
}
