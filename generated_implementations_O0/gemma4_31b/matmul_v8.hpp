#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Optimized Matrix Multiplication with Packed Binary Matrix B
// Combining v2's loop structure with v3's 8-bit lookup table.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    
    // Precompute sign table for 8-bit patterns.
    // Each 8-bit pattern maps to two float32x4_t vectors.
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
        // K is max 3072, so K/4 = 768.
        float32x4_t accs[768]; 
        for (size_t v = 0; v < K / 4; ++v) {
            accs[v] = vdupq_n_f32(0.0f);
        }

        for (size_t p = 0; p < K; ++p) {
            float a_val = row_A[p];
            if (a_val == 0.0f) continue;
            
            float32x4_t v_a = vdupq_n_f32(a_val);
            const uint32_t* row_B_packed = B + p * K_ints;

            for (size_t b = 0; b < K_ints; ++b) {
                uint32_t packed = row_B_packed[b];
                if (packed == 0 && a_val != 0.0f) {
                    // Even if packed is 0, we must consider the -1.0f signs.
                    // Wait, if packed is 0, all signs are -1.0f.
                    // But our lookup table handles the signs.
                    // If packed is 0, b0=0, b1=0, b2=0, b3=0.
                    // sign_table[0] will contain [-1, -1, -1, -1].
                }
                
                uint8_t b0 = (uint8_t)(packed & 0xFF);
                uint8_t b1 = (uint8_t)((packed >> 8) & 0xFF);
                uint8_t b2 = (uint8_t)((packed >> 16) & 0xFF);
                uint8_t b3 = (uint8_t)((packed >> 24) & 0xFF);

                size_t base_idx = b * 8;
                accs[base_idx + 0] = vmlaq_f32(accs[base_idx + 0], v_a, sign_table[2 * b0]);
                accs[base_idx + 1] = vmlaq_f32(accs[base_idx + 1], v_a, sign_table[2 * b0 + 1]);
                accs[base_idx + 2] = vmlaq_f32(accs[base_idx + 2], v_a, sign_table[2 * b1]);
                accs[base_idx + 3] = vmlaq_f32(accs[base_idx + 3], v_a, sign_table[2 * b1 + 1]);
                accs[base_idx + 4] = vmlaq_f32(accs[base_idx + 4], v_a, sign_table[2 * b2]);
                accs[base_idx + 5] = vmlaq_f32(accs[base_idx + 5], v_a, sign_table[2 * b2 + 1]);
                accs[base_idx + 6] = vmlaq_f32(accs[base_idx + 6], v_a, sign_table[2 * b3]);
                accs[base_idx + 7] = vmlaq_f32(accs[base_idx + 7], v_a, sign_table[2 * b3 + 1]);
            }
        }

        for (size_t v = 0; v < K / 4; ++v) {
            vst1q_f32(row_C + v * 4, accs[v]);
        }
    }
}
