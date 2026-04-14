#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Optimized Matrix C = Matrix A * Matrix B
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    // Precompute sign lookup table for 4-bit patterns
    float32x4_t sign_lut[16];
    for (int i = 0; i < 16; ++i) {
        float s[4];
        for (int b = 0; b < 4; ++b) {
            s[b] = ((i >> b) & 1) ? 1.0f : -1.0f;
        }
        sign_lut[i] = vld1q_f32(s);
    }

    for (size_t i = 0; i < M; ++i) {
        const float* rowA = &A[i * K];
        float* rowC = &C[i * K];

        // Tiling the j_int dimension to improve cache locality of B and reuse rowA
        for (size_t j_block = 0; j_block < K_ints; j_block += 2) {
            float32x4_t acc00 = vdupq_n_f32(0.0f), acc01 = vdupq_n_f32(0.0f), acc02 = vdupq_n_f32(0.0f), acc03 = vdupq_n_f32(0.0f),
                         acc04 = vdupq_n_f32(0.0f), acc05 = vdupq_n_f32(0.0f), acc06 = vdupq_n_f32(0.0f), acc07 = vdupq_n_f32(0.0f),
                         acc10 = vdupq_n_f32(0.0f), acc11 = vdupq_n_f32(0.0f), acc12 = vdupq_n_f32(0.0f), acc13 = vdupq_n_f32(0.0f),
                         acc14 = vdupq_n_f32(0.0f), acc15 = vdupq_n_f32(0.0f), acc16 = vdupq_n_f32(0.0f), acc17 = vdupq_n_f32(0.0f);

            for (size_t p = 0; p < K; ++p) {
                const float32x4_t v_a = vdupq_n_f32(rowA[p]);
                const uint32_t packed0 = B[p * K_ints + j_block];
                
                acc00 = vmlaq_f32(acc00, sign_lut[(packed0 >> 0) & 0xF], v_a);
                acc01 = vmlaq_f32(acc01, sign_lut[(packed0 >> 4) & 0xF], v_a);
                acc02 = vmlaq_f32(acc02, sign_lut[(packed0 >> 8) & 0xF], v_a);
                acc03 = vmlaq_f32(acc03, sign_lut[(packed0 >> 12) & 0xF], v_a);
                acc04 = vmlaq_f32(acc04, sign_lut[(packed0 >> 16) & 0xF], v_a);
                acc05 = vmlaq_f32(acc05, sign_lut[(packed0 >> 20) & 0xF], v_a);
                acc06 = vmlaq_f32(acc06, sign_lut[(packed0 >> 24) & 0xF], v_a);
                acc07 = vmlaq_f32(acc07, sign_lut[(packed0 >> 28) & 0xF], v_a);

                if (j_block + 1 < K_ints) {
                    const uint32_t packed1 = B[p * K_ints + j_block + 1];
                    acc10 = vmlaq_f32(acc10, sign_lut[(packed1 >> 0) & 0xF], v_a);
                    acc11 = vmlaq_f32(acc11, sign_lut[(packed1 >> 4) & 0xF], v_a);
                    acc12 = vmlaq_f32(acc12, sign_lut[(packed1 >> 8) & 0xF], v_a);
                    acc13 = vmlaq_f32(acc13, sign_lut[(packed1 >> 12) & 0xF], v_a);
                    acc14 = vmlaq_f32(acc14, sign_lut[(packed1 >> 16) & 0xF], v_a);
                    acc15 = vmlaq_f32(acc15, sign_lut[(packed1 >> 20) & 0xF], v_a);
                    acc16 = vmlaq_f32(acc16, sign_lut[(packed1 >> 24) & 0xF], v_a);
                    acc17 = vmlaq_f32(acc17, sign_lut[(packed1 >> 28) & 0xF], v_a);
                }
            }
            
            float* out_ptr0 = &rowC[j_block * 32];
            vst1q_f32(out_ptr0, acc00);
            vst1q_f32(out_ptr0 + 4, acc01);
            vst1q_f32(out_ptr0 + 8, acc02);
            vst1q_f32(out_ptr0 + 12, acc03);
            vst1q_f32(out_ptr0 + 16, acc04);
            vst1q_f32(out_ptr0 + 20, acc05);
            vst1q_f32(out_ptr0 + 24, acc06);
            vst1q_f32(out_ptr0 + 28, acc07);
            if (j_block + 1 < K_ints) {
                float* out_ptr1 = &rowC[(j_block + 1) * 32];
                vst1q_f32(out_ptr1, acc10);
                vst1q_f32(out_ptr1 + 4, acc11);
                vst1q_f32(out_ptr1 + 8, acc12);
                vst1q_f32(out_ptr1 + 12, acc13);
                vst1q_f32(out_ptr1 + 16, acc14);
                vst1q_f32(out_ptr1 + 20, acc15);
                vst1q_f32(out_ptr1 + 24, acc16);
                vst1q_f32(out_ptr1 + 28, acc17);
            }
        }
    }
}
