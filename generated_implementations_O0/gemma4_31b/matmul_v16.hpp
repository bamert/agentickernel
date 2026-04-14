#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Optimized Matrix Multiplication with Packed Binary Matrix B
// This version refines matmul_v13 by using a slightly smaller M_BLOCK
// to allow more space in L1 and manually unrolling the inner loops.
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

    const size_t M_BLOCK = 2;
    // L1 Cache size: 64 KB. 2 * 768 * 16 bytes = 24.5 KB.
    float32x4_t accs[M_BLOCK][768];

    for (size_t i_start = 0; i_start < M; i_start += M_BLOCK) {
        size_t i_end = (i_start + M_BLOCK < M) ? (i_start + M_BLOCK) : M;
        
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
            float32x4_t* p_acc0 = accs[0];
            float32x4_t* p_acc1 = accs[1];

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

                const float32x4_t va0 = v_a[0];
                const float32x4_t va1 = v_a[1];

                p_acc0[0] = vmlaq_f32(p_acc0[0], va0, s0);
                p_acc1[0] = vmlaq_f32(p_acc1[0], va1, s0);
                p_acc0[1] = vmlaq_f32(p_acc0[1], va0, s1);
                p_acc1[1] = vmlaq_f32(p_acc1[1], va1, s1);
                p_acc0[2] = vmlaq_f32(p_acc0[2], va0, s2);
                p_acc1[2] = vmlaq_f32(p_acc1[2], va1, s2);
                p_acc0[3] = vmlaq_f32(p_acc0[3], va0, s3);
                p_acc1[3] = vmlaq_f32(p_acc1[3], va1, s3);
                p_acc0[4] = vmlaq_f32(p_acc0[4], va0, s4);
                p_acc1[4] = vmlaq_f32(p_acc1[4], va1, s4);
                p_acc0[5] = vmlaq_f32(p_acc0[5], va0, s5);
                p_acc1[5] = vmlaq_f32(p_acc1[5], va1, s5);
                p_acc0[6] = vmlaq_f32(p_acc0[6], va0, s6);
                p_acc1[6] = vmlaq_f32(p_acc1[6], va1, s6);
                p_acc0[7] = vmlaq_f32(p_acc0[7], va0, s7);
                p_acc1[7] = vmlaq_f32(p_acc1[7], va1, s7);
                
                p_acc0 += 8;
                p_acc1 += 8;
            }
        }

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
