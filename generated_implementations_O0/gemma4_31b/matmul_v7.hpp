#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Optimized Matrix Multiplication with Packed Binary Matrix B
// Strategy:
// 1. Use a precomputed sign lookup table for 8-bit patterns.
// 2. Change loop order to maximize reuse of the B matrix and sign lookups.
// 3. Process C in blocks of columns (K_BLOCK) to keep accumulators in L1.
// 4. Use the loop order: j_start -> p -> i, so that sign vectors are loaded once 
//    and reused for all M rows of A.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    
    // Precompute sign table for 8-bit patterns.
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

    const size_t K_BLOCK = 256; // Process 256 columns at a time.
    
    // Temporarily store results in a buffer to avoid repeated slow stores to C.
    // This buffer stores M rows, each with K_BLOCK / 4 vectors.
    // 32 * 64 * 16 bytes = 32 KB. Fits in L1.
    float32x4_t accs[32][64];

    for (size_t j_start = 0; j_start < K; j_start += K_BLOCK) {
        size_t j_end = (j_start + K_BLOCK < K) ? (j_start + K_BLOCK) : K;
        size_t current_block_vecs = (j_end - j_start) / 4;
        size_t b_start = j_start / 32;
        size_t b_end = j_end / 32;

        // Initialize accumulators
        for (size_t i = 0; i < M; ++i) {
            for (size_t v = 0; v < current_block_vecs; ++v) {
                accs[i][v] = vdupq_n_f32(0.0f);
            }
        }

        for (size_t p = 0; p < K; ++p) {
            const uint32_t* row_B_ptr = B + p * K_ints;
            
            for (size_t b = b_start; b < b_end; ++b) {
                uint32_t packed = row_B_ptr[b];
                
                uint8_t b0 = (uint8_t)(packed & 0xFF);
                uint8_t b1 = (uint8_t)((packed >> 8) & 0xFF);
                uint8_t b2 = (uint8_t)((packed >> 16) & 0xFF);
                uint8_t b3 = (uint8_t)((packed >> 24) & 0xFF);

                float32x4_t s0 = sign_table[2 * b0];
                float32x4_t s1 = sign_table[2 * b0 + 1];
                float32x4_t s2 = sign_table[2 * b1];
                float32x4_t s3 = sign_table[2 * b1 + 1];
                float32x4_t s4 = sign_table[2 * b2];
                float32x4_t s5 = sign_table[2 * b2 + 1];
                float32x4_t s6 = sign_table[2 * b3];
                float32x4_t s7 = sign_table[2 * b3 + 1];

                size_t base_idx = (b - b_start) * 8;
                
                for (size_t i = 0; i < M; ++i) {
                    float a_val = A[i * K + p];
                    float32x4_t v_a = vdupq_n_f32(a_val);
                    
                    float32x4_t* row_acc = accs[i];
                    row_acc[base_idx + 0] = vmlaq_f32(row_acc[base_idx + 0], v_a, s0);
                    row_acc[base_idx + 1] = vmlaq_f32(row_acc[base_idx + 1], v_a, s1);
                    row_acc[base_idx + 2] = vmlaq_f32(row_acc[base_idx + 2], v_a, s2);
                    row_acc[base_idx + 3] = vmlaq_f32(row_acc[base_idx + 3], v_a, s3);
                    row_acc[base_idx + 4] = vmlaq_f32(row_acc[base_idx + 4], v_a, s4);
                    row_acc[base_idx + 5] = vmlaq_f32(row_acc[base_idx + 5], v_a, s5);
                    row_acc[base_idx + 6] = vmlaq_f32(row_acc[base_idx + 6], v_a, s6);
                    row_acc[base_idx + 7] = vmlaq_f32(row_acc[base_idx + 7], v_a, s7);
                }
            }
        }

        // Store block results
        for (size_t i = 0; i < M; ++i) {
            float* row_C = C + i * K + j_start;
            for (size_t v = 0; v < current_block_vecs; ++v) {
                vst1q_f32(row_C + v * 4, accs[i][v]);
            }
        }
    }
}
