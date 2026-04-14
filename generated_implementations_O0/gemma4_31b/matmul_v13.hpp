#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Optimized Matrix Multiplication with Packed Binary Matrix B
// This version combines the pointer-based inner loop of v11 with the 
// sign-reuse strategy (M_BLOCK) to reduce memory lookups.
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

    const size_t M_BLOCK = 4;
    // Accumulators for M_BLOCK rows. 
    // 4 * 768 * 16 bytes = 49 KB. Fits in L1 (64 KB).
    float32x4_t accs[M_BLOCK][768];

    for (size_t i_start = 0; i_start < M; i_start += M_BLOCK) {
        size_t i_end = (i_start + M_BLOCK < M) ? (i_start + M_BLOCK) : M;
        
        // Reset accumulators for this block of rows
        for (size_t r = 0; r < M_BLOCK; ++r) {
            for (size_t v = 0; v < K / 4; ++v) {
                accs[r][v] = vdupq_n_f32(0.0f);
            }
        }

        for (size_t p = 0; p < K; ++p) {
            float32x4_t v_a[M_BLOCK];
            bool any_nonzero = false;
            for (size_t r = 0; r < M_BLOCK; ++r) {
                if (i_start + r < i_end) {
                    float val = A[(i_start + r) * K + p];
                    v_a[r] = vdupq_n_f32(val);
                    if (val != 0.0f) any_nonzero = true;
                } else {
                    v_a[r] = vdupq_n_f32(0.0f);
                }
            }

            if (!any_nonzero) continue;

            const uint32_t* row_B_packed = B + p * K_ints;
            
            // Pointers to the start of each row's accumulators
            float32x4_t* p_acc[M_BLOCK];
            for (size_t r = 0; r < M_BLOCK; ++r) {
                p_acc[r] = accs[r];
            }

            for (size_t b = 0; b < K_ints; ++b) {
                const uint32_t packed = row_B_packed[b];
                
                const uint8_t b0 = (uint8_t)(packed & 0xFF);
                const uint8_t b1 = (uint8_t)((packed >> 8) & 0xFF);
                const uint8_t b2 = (uint8_t)((packed >> 16) & 0xFF);
                const uint8_t b3 = (uint8_t)((packed >> 24) & 0xFF);

                const float32x4_t s0 = sign_table[2 * b0];
                const float32x4_t s1 = sign_table[2 * b0 + 1];
                const float32x4_t s2 = sign_table[2 * b1];
                const float32x4_t s3 = sign_table[2 * b1 + 1];
                const float32x4_t s4 = sign_table[2 * b2];
                const float32x4_t s5 = sign_table[2 * b2 + 1];
                const float32x4_t s6 = sign_table[2 * b3];
                const float32x4_t s7 = sign_table[2 * b3 + 1];

                for (size_t r = 0; r < M_BLOCK; ++r) {
                    float32x4_t* cur_acc = p_acc[r];
                    const float32x4_t va = v_a[r];
                    cur_acc[0] = vmlaq_f32(cur_acc[0], va, s0);
                    cur_acc[1] = vmlaq_f32(cur_acc[1], va, s1);
                    cur_acc[2] = vmlaq_f32(cur_acc[2], va, s2);
                    cur_acc[3] = vmlaq_f32(cur_acc[3], va, s3);
                    cur_acc[4] = vmlaq_f32(cur_acc[4], va, s4);
                    cur_acc[5] = vmlaq_f32(cur_acc[5], va, s5);
                    cur_acc[6] = vmlaq_f32(cur_acc[6], va, s6);
                    cur_acc[7] = vmlaq_f32(cur_acc[7], va, s7);
                    p_acc[r] += 8;
                }
            }
        }

        // Store results back to C
        for (size_t r = 0; r < M_BLOCK; ++r) {
            if (i_start + r < i_end) {
                float* row_C = C + (i_start + r) * K;
                for (size_t v = 0; v < K / 4; ++v) {
                    vst1q_f32(row_C + v * 4, accs[r][v]);
                }
            }
        }
    }
}
