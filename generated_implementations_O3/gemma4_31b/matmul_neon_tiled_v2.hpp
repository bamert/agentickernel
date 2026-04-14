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

        for (size_t j_block = 0; j_block < K_ints; j_block += 4) {
            float32x4_t acc00 = vdupq_n_f32(0.0f), acc01 = vdupq_n_f32(0.0f), acc02 = vdupq_n_f32(0.0f), acc03 = vdupq_n_f32(0.0f),
                         acc04 = vdupq_n_f32(0.0f), acc05 = vdupq_n_f32(0.0f), acc06 = vdupq_n_f32(0.0f), acc07 = vdupq_n_f32(0.0f),
                         acc10 = vdupq_n_f32(0.0f), acc11 = vdupq_n_f32(0.0f), acc12 = vdupq_n_f32(0.0f), acc13 = vdupq_n_f32(0.0f),
                         acc14 = vdupq_n_f32(0.0f), acc15 = vdupq_n_f32(0.0f), acc16 = vdupq_n_f32(0.0f), acc17 = vdupq_n_f32(0.0f),
                         acc20 = vdupq_n_f32(0.0f), acc21 = vdupq_n_f32(0.0f), acc22 = vdupq_n_f32(0.0f), acc23 = vdupq_n_f32(0.0f),
                         acc24 = vdupq_n_f32(0.0f), acc25 = vdupq_n_f32(0.0f), acc26 = vdupq_n_f32(0.0f), acc27 = vdupq_n_f32(0.0f),
                         acc30 = vdupq_n_f32(0.0f), acc31 = vdupq_n_f32(0.0f), acc32 = vdupq_n_f32(0.0f), acc33 = vdupq_n_f32(0.0f),
                         acc34 = vdupq_n_f32(0.0f), acc35 = vdupq_n_f32(0.0f), acc36 = vdupq_n_f32(0.0f), acc37 = vdupq_n_f32(0.0f);

            for (size_t p = 0; p < K; ++p) {
                const float32x4_t v_a = vdupq_n_f32(rowA[p]);
                
                const uint32_t p0 = B[p * K_ints + j_block];
                acc00 = vmlaq_f32(acc00, sign_lut[(p0 >> 0) & 0xF], v_a);
                acc01 = vmlaq_f32(acc01, sign_lut[(p0 >> 4) & 0xF], v_a);
                acc02 = vmlaq_f32(acc02, sign_lut[(p0 >> 8) & 0xF], v_a);
                acc03 = vmlaq_f32(acc03, sign_lut[(p0 >> 12) & 0xF], v_a);
                acc04 = vmlaq_f32(acc04, sign_lut[(p0 >> 16) & 0xF], v_a);
                acc05 = vmlaq_f32(acc05, sign_lut[(p0 >> 20) & 0xF], v_a);
                acc06 = vmlaq_f32(acc06, sign_lut[(p0 >> 24) & 0xF], v_a);
                acc07 = vmlaq_f32(acc07, sign_lut[(p0 >> 28) & 0xF], v_a);

                if (j_block + 1 < K_ints) {
                    const uint32_t p1 = B[p * K_ints + j_block + 1];
                    acc10 = vmlaq_f32(acc10, sign_lut[(p1 >> 0) & 0xF], v_a);
                    acc11 = vmlaq_f32(acc11, sign_lut[(p1 >> 4) & 0xF], v_a);
                    acc12 = vmlaq_f32(acc12, sign_lut[(p1 >> 8) & 0xF], v_a);
                    acc13 = vmlaq_f32(acc13, sign_lut[(p1 >> 12) & 0xF], v_a);
                    acc14 = vmlaq_f32(acc14, sign_lut[(p1 >> 16) & 0xF], v_a);
                    acc15 = vmlaq_f32(acc15, sign_lut[(p1 >> 20) & 0xF], v_a);
                    acc16 = vmlaq_f32(acc16, sign_lut[(p1 >> 24) & 0xF], v_a);
                    acc17 = vmlaq_f32(acc17, sign_lut[(p1 >> 28) & 0xF], v_a);
                }
                if (j_block + 2 < K_ints) {
                    const uint32_t p2 = B[p * K_ints + j_block + 2];
                    acc20 = vmlaq_f32(acc20, sign_lut[(p2 >> 0) & 0xF], v_a);
                    acc21 = vmlaq_f32(acc21, sign_lut[(p2 >> 4) & 0xF], v_a);
                    acc22 = vmlaq_f32(acc22, sign_lut[(p2 >> 8) & 0xF], v_a);
                    acc23 = vmlaq_f32(acc23, sign_lut[(p2 >> 12) & 0xF], v_a);
                    acc24 = vmlaq_f32(acc24, sign_lut[(p2 >> 16) & 0xF], v_a);
                    acc25 = vmlaq_f32(acc25, sign_lut[(p2 >> 20) & 0xF], v_a);
                    acc26 = vmlaq_f32(acc26, sign_lut[(p2 >> 24) & 0xF], v_a);
                    acc27 = vmlaq_f32(acc27, sign_lut[(p2 >> 28) & 0xF], v_a);
                }
                if (j_block + 3 < K_ints) {
                    const uint32_t p3 = B[p * K_ints + j_block + 3];
                    acc30 = vmlaq_f32(acc30, sign_lut[(p3 >> 0) & 0xF], v_a);
                    acc31 = vmlaq_f32(acc31, sign_lut[(p3 >> 4) & 0xF], v_a);
                    acc32 = vmlaq_f32(acc32, sign_lut[(p3 >> 8) & 0xF], v_a);
                    acc33 = vmlaq_f32(acc33, sign_lut[(p3 >> 12) & 0xF], v_a);
                    acc34 = vmlaq_f32(acc34, sign_lut[(p3 >> 16) & 0xF], v_a);
                    acc35 = vmlaq_f32(acc35, sign_lut[(p3 >> 20) & 0xF], v_a);
                    acc36 = vmlaq_f32(acc36, sign_lut[(p3 >> 24) & 0xF], v_a);
                    acc37 = vmlaq_f32(acc37, sign_lut[(p3 >> 28) & 0xF], v_a);
                }
            }
            
            float* out_ptr = &rowC[j_block * 32];
            vst1q_f32(out_ptr, acc00);
            vst1q_f32(out_ptr + 4, acc01);
            vst1q_f32(out_ptr + 8, acc02);
            vst1q_f32(out_ptr + 12, acc03);
            vst1q_f32(out_ptr + 16, acc04);
            vst1q_f32(out_ptr + 20, acc05);
            vst1q_f32(out_ptr + 24, acc06);
            vst1q_f32(out_ptr + 28, acc07);
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
            if (j_block + 2 < K_ints) {
                float* out_ptr2 = &rowC[(j_block + 2) * 32];
                vst1q_f32(out_ptr2, acc20);
                vst1q_f32(out_ptr2 + 4, acc21);
                vst1q_f32(out_ptr2 + 8, acc22);
                vst1q_f32(out_ptr2 + 12, acc23);
                vst1q_f32(out_ptr2 + 16, acc24);
                vst1q_f32(out_ptr2 + 20, acc25);
                vst1q_f32(out_ptr2 + 24, acc26);
                vst1q_f32(out_ptr2 + 28, acc27);
            }
            if (j_block + 3 < K_ints) {
                float* out_ptr3 = &rowC[(j_block + 3) * 32];
                vst1q_f32(out_ptr3, acc30);
                vst1q_f32(out_ptr3 + 4, acc31);
                vst1q_f32(out_ptr3 + 8, acc32);
                vst1q_f32(out_ptr3 + 12, acc33);
                vst1q_f32(out_ptr3 + 16, acc34);
                vst1q_f32(out_ptr3 + 20, acc35);
                vst1q_f32(out_ptr3 + 24, acc36);
                vst1q_f32(out_ptr3 + 28, acc37);
            }
        }
    }
}
