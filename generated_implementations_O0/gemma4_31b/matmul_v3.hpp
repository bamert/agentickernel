#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Optimized Matrix Multiplication with Packed Binary Matrix B
// Logic: C[i][j] = Sum_{p=0..K-1} A[i][p] * (B[p][j] ? 1.0f : -1.0f)
// 
// Improvements over v2:
// 1. Use an 8-bit lookup table to reduce the number of lookups per uint32_t.
// 2. Use a blocking factor for K to ensure the accumulator buffer fits in L1 and 
//    handles larger K values than v2.
// 3. Add zero-check for A[i][p] to skip computations.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    
    // Precompute sign table for 8-bit patterns.
    // Each 8-bit pattern maps to two float32x4_t vectors (8 floats total).
    // Table size: 256 patterns * 2 vectors/pattern * 16 bytes/vector = 8192 bytes.
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

    const size_t K_BLOCK = 512;
    const size_t BLOCK_VECS = K_BLOCK / 4;
    float32x4_t accs[128]; // 512 / 4 = 128 vectors

    for (size_t i = 0; i < M; ++i) {
        const float* row_A = A + i * K;
        float* row_C = C + i * K;

        for (size_t j_start = 0; j_start < K; j_start += K_BLOCK) {
            size_t j_end = (j_start + K_BLOCK < K) ? (j_start + K_BLOCK) : K;
            size_t current_block_vecs = (j_end - j_start) / 4;
            size_t b_start = j_start / 32;
            size_t b_end = j_end / 32;

            // Initialize accumulators for this block
            for (size_t v = 0; v < current_block_vecs; ++v) {
                accs[v] = vdupq_n_f32(0.0f);
            }

            for (size_t p = 0; p < K; ++p) {
                float a_val = row_A[p];
                if (a_val == 0.0f) continue;
                
                float32x4_t v_a = vdupq_n_f32(a_val);
                const uint32_t* row_B_packed = B + p * K_ints;

                for (size_t b = b_start; b < b_end; ++b) {
                    uint32_t packed = row_B_packed[b];
                    
                    // Extract 4 bytes and use them as indices into the 8-bit lookup table
                    uint8_t b0 = (uint8_t)(packed & 0xFF);
                    uint8_t b1 = (uint8_t)((packed >> 8) & 0xFF);
                    uint8_t b2 = (uint8_t)((packed >> 16) & 0xFF);
                    uint8_t b3 = (uint8_t)((packed >> 24) & 0xFF);

                    size_t base_idx = (b - b_start) * 8;
                    
                    // Each byte indexing provides 2 vectors of signs
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

            // Store results back to C
            for (size_t v = 0; v < current_block_vecs; ++v) {
                vst1q_f32(row_C + j_start + v * 4, accs[v]);
            }
        }
    }
}
