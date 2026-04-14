#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Optimized Matrix Multiplication with Packed Binary Matrix B
// This version builds upon v8 by unrolling the inner loop and 
// optimizing for instruction-level parallelism.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    
    // Precompute sign table for 8-bit patterns.
    // sign_table[2 * m] and [2 * m + 1] store the 8 signs for pattern m.
    float32x4_t sign_table[512];
    for (int m = 0; m < 256; ++m) {
        float signs_low[4], signs_high[4];
        for (int b = 0; b < 4; ++b) {
            signs_low[b] = (m & (1 << b)) ? 1.0f : -1.0f;
            signs_high[b] = (m & (1 << (b + 4))) ? 1.0f : -1.0f;
        }
        sign_table[2 * m] = vld1q_f32(signs_low);
        sign_table[2 * m + 1] = vld1q_f32(signs_high);
    }

    for (size_t i = 0; i < M; ++i) {
        const float* row_A = A + i * K;
        float* row_C = C + i * K;

        // Accumulators for the entire row of C.
        // K <= 3072, so K/4 <= 768. 
        // Using a fixed-size buffer aligned to 16 bytes.
        float32x4_t accs[768]; 
        for (size_t v = 0; v < K / 4; ++v) {
            accs[v] = vdupq_n_f32(0.0f);
        }

        for (size_t p = 0; p < K; ++p) {
            float a_val = row_A[p];
            if (a_val == 0.0f) continue;
            
            float32x4_t v_a = vdupq_n_f32(a_val);
            const uint32_t* row_B_packed = B + p * K_ints;

            // Unroll the inner loop to improve pipeline utilization.
            for (size_t b = 0; b < K_ints; b += 2) {
                uint32_t packed0 = row_B_packed[b];
                uint32_t packed1 = row_B_packed[b + 1];
                
                uint8_t b0_0 = (uint8_t)(packed0 & 0xFF);
                uint8_t b0_1 = (uint8_t)((packed0 >> 8) & 0xFF);
                uint8_t b0_2 = (uint8_t)((packed0 >> 16) & 0xFF);
                uint8_t b0_3 = (uint8_t)((packed0 >> 24) & 0xFF);

                uint8_t b1_0 = (uint8_t)(packed1 & 0xFF);
                uint8_t b1_1 = (uint8_t)((packed1 >> 8) & 0xFF);
                uint8_t b1_2 = (uint8_t)((packed1 >> 16) & 0xFF);
                uint8_t b1_3 = (uint8_t)((packed1 >> 24) & 0xFF);

                size_t base0 = b * 8;
                size_t base1 = (b + 1) * 8;

                accs[base0 + 0] = vmlaq_f32(accs[base0 + 0], v_a, sign_table[2 * b0_0]);
                accs[base0 + 1] = vmlaq_f32(accs[base0 + 1], v_a, sign_table[2 * b0_0 + 1]);
                accs[base0 + 2] = vmlaq_f32(accs[base0 + 2], v_a, sign_table[2 * b0_1]);
                accs[base0 + 3] = vmlaq_f32(accs[base0 + 3], v_a, sign_table[2 * b0_1 + 1]);
                accs[base0 + 4] = vmlaq_f32(accs[base0 + 4], v_a, sign_table[2 * b0_2]);
                accs[base0 + 5] = vmlaq_f32(accs[base0 + 5], v_a, sign_table[2 * b0_2 + 1]);
                accs[base0 + 6] = vmlaq_f32(accs[base0 + 6], v_a, sign_table[2 * b0_3]);
                accs[base0 + 7] = vmlaq_f32(accs[base0 + 7], v_a, sign_table[2 * b0_3 + 1]);

                accs[base1 + 0] = vmlaq_f32(accs[base1 + 0], v_a, sign_table[2 * b1_0]);
                accs[base1 + 1] = vmlaq_f32(accs[base1 + 1], v_a, sign_table[2 * b1_0 + 1]);
                accs[base1 + 2] = vmlaq_f32(accs[base1 + 2], v_a, sign_table[2 * b1_1]);
                accs[base1 + 3] = vmlaq_f32(accs[base1 + 3], v_a, sign_table[2 * b1_1 + 1]);
                accs[base1 + 4] = vmlaq_f32(accs[base1 + 4], v_a, sign_table[2 * b1_2]);
                accs[base1 + 5] = vmlaq_f32(accs[base1 + 5], v_a, sign_table[2 * b1_2 + 1]);
                accs[base1 + 6] = vmlaq_f32(accs[base1 + 6], v_a, sign_table[2 * b1_3]);
                accs[base1 + 7] = vmlaq_f32(accs[base1 + 7], v_a, sign_table[2 * b1_3 + 1]);
            }
        }

        for (size_t v = 0; v < K / 4; ++v) {
            vst1q_f32(row_C + v * 4, accs[v]);
        }
    }
}
