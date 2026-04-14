#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Optimized Matrix Multiplication with Packed Binary Matrix B
// This version prioritizes the reuse of B matrix elements.
// Since B is the largest structure and M is small, we process multiple rows of A
// and blocks of C to maximize B's cache residency.
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

    // Block sizes to balance register usage and L1 cache.
    const size_t M_BLOCK = 4;
    const size_t K_BLOCK = 256; // 8 uint32's
    const size_t VECS_PER_BLOCK = K_BLOCK / 4; // 64 vectors

    for (size_t i_start = 0; i_start < M; i_start += M_BLOCK) {
        size_t i_end = (i_start + M_BLOCK < M) ? (i_start + M_BLOCK) : M;
        
        for (size_t j_start = 0; j_start < K; j_start += K_BLOCK) {
            size_t j_end = (j_start + K_BLOCK < K) ? (j_start + K_BLOCK) : K;
            size_t current_block_vecs = (j_end - j_start) / 4;
            size_t b_start = j_start / 32;
            size_t b_end = j_end / 32;

            // Accumulators for the current row-block and col-block.
            // M_BLOCK * 64 vectors * 16 bytes = 4KB.
            float32x4_t accs[M_BLOCK][64];
            for (size_t r = 0; r < M_BLOCK; ++r) {
                for (size_t v = 0; v < current_block_vecs; ++v) {
                    accs[r][v] = vdupq_n_f32(0.0f);
                }
            }

            for (size_t p = 0; p < K; ++p) {
                // Load A values for the row block once per p.
                float32x4_t v_a[M_BLOCK];
                for (size_t r = 0; r < M_BLOCK; ++r) {
                    if (i_start + r < i_end) {
                        v_a[r] = vdupq_n_f32(A[(i_start + r) * K + p]);
                    } else {
                        v_a[r] = vdupq_n_f32(0.0f);
                    }
                }

                const uint32_t* row_B_packed = B + p * K_ints;

                for (size_t b = b_start; b < b_end; ++b) {
                    uint32_t packed = row_B_packed[b];
                    
                    // 8-bit lookup for signs
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
                    
                    // Update accumulators for all rows in the M_BLOCK.
                    for (size_t r = 0; r < M_BLOCK; ++r) {
                        float32x4_t va = v_a[r];
                        accs[r][base_idx + 0] = vmlaq_f32(accs[r][base_idx + 0], va, s0);
                        accs[r][base_idx + 1] = vmlaq_f32(accs[r][base_idx + 1], va, s1);
                        accs[r][base_idx + 2] = vmlaq_f32(accs[r][base_idx + 2], va, s2);
                        accs[r][base_idx + 3] = vmlaq_f32(accs[r][base_idx + 3], va, s3);
                        accs[r][base_idx + 4] = vmlaq_f32(accs[r][base_idx + 4], va, s4);
                        accs[r][base_idx + 5] = vmlaq_f32(accs[r][base_idx + 5], va, s5);
                        accs[r][base_idx + 6] = vmlaq_f32(accs[r][base_idx + 6], va, s6);
                        accs[r][base_idx + 7] = vmlaq_f32(accs[r][base_idx + 7], va, s7);
                    }
                }
            }

            // Store results
            for (size_t r = 0; r < M_BLOCK; ++r) {
                if (i_start + r < i_end) {
                    float* row_C = C + (i_start + r) * K + j_start;
                    for (size_t v = 0; v < current_block_vecs; ++v) {
                        vst1q_f32(row_C + v * 4, accs[r][v]);
                    }
                }
            }
        }
    }
}
