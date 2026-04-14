#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Optimized Matrix Multiplication with Packed Binary Matrix B
// Logic: C[i][j] = Sum_{p=0..K-1} A[i][p] * (B[p][j] ? 1.0f : -1.0f)
// 
// Optimization: 
// 1. Use NEON SIMD to compute 4 columns of C at once.
// 2. Precompute a SIGN_TABLE that maps a 4-bit integer (0-15) to a float32x4_t 
//    containing the corresponding signs (1.0f for 1, -1.0f for 0).
// 3. Process the outer loops to maintain good cache locality.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    
    // Precompute sign table for 4-bit patterns
    // SIGN_TABLE[m] stores the signs for bits of integer m
    float32x4_t sign_table[16];
    for (int m = 0; m < 16; ++m) {
        float signs[4];
        for (int b = 0; b < 4; ++b) {
            signs[b] = (m & (1 << b)) ? 1.0f : -1.0f;
        }
        sign_table[m] = vld1q_f32(signs);
    }

    for (size_t i = 0; i < M; ++i) {
        const float* row_A = A + i * K;
        float* row_C = C + i * K;

        // We use a temporary buffer to store accumulators for one row of C
        // Row C is K floats. We process them in chunks of 4.
        // K is a multiple of 32, so K/4 is consistent.
        float32x4_t accs[768]; // K_max = 3072, 3072/4 = 768
        
        // Initialize accumulators to zero
        for (size_t j = 0; j < K / 4; ++j) {
            accs[j] = vdupq_n_f32(0.0f);
        }

        for (size_t p = 0; p < K; ++p) {
            float a_val = row_A[p];
            float32x4_t v_a = vdupq_n_f32(a_val);
            const uint32_t* row_B_packed = B + p * K_ints;

            for (size_t b = 0; b < K_ints; ++b) {
                uint32_t packed = row_B_packed[b];
                
                // Process 32 bits in 8 chunks of 4 bits
                // chunk 0: bits 0-3, chunk 1: bits 4-7, ...
                uint32_t bits_0_3 = packed & 0xF;
                uint32_t bits_4_7 = (packed >> 4) & 0xF;
                uint32_t bits_8_11 = (packed >> 8) & 0xF;
                uint32_t bits_12_15 = (packed >> 12) & 0xF;
                uint32_t bits_16_19 = (packed >> 16) & 0xF;
                uint32_t bits_20_23 = (packed >> 20) & 0xF;
                uint32_t bits_24_27 = (packed >> 24) & 0xF;
                uint32_t bits_28_31 = (packed >> 28) & 0xF;

                size_t base_idx = b * 8;
                accs[base_idx + 0] = vmlaq_f32(accs[base_idx + 0], v_a, sign_table[bits_0_3]);
                accs[base_idx + 1] = vmlaq_f32(accs[base_idx + 1], v_a, sign_table[bits_4_7]);
                accs[base_idx + 2] = vmlaq_f32(accs[base_idx + 2], v_a, sign_table[bits_8_11]);
                accs[base_idx + 3] = vmlaq_f32(accs[base_idx + 3], v_a, sign_table[bits_12_15]);
                accs[base_idx + 4] = vmlaq_f32(accs[base_idx + 4], v_a, sign_table[bits_16_19]);
                accs[base_idx + 5] = vmlaq_f32(accs[base_idx + 5], v_a, sign_table[bits_20_23]);
                accs[base_idx + 6] = vmlaq_f32(accs[base_idx + 6], v_a, sign_table[bits_24_27]);
                accs[base_idx + 7] = vmlaq_f32(accs[base_idx + 7], v_a, sign_table[bits_28_31]);
            }
        }

        // Store results back to C
        for (size_t j = 0; j < K / 4; ++j) {
            vst1q_f32(row_C + j * 4, accs[j]);
        }
    }
}
