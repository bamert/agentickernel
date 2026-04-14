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

    struct SignEntry {
        float32x4_t v0;
        float32x4_t v1;
    };
    SignEntry sign_lut[256];

    for (int i = 0; i < 256; ++i) {
        float s0[4], s1[4];
        for (int b = 0; b < 4; ++b) {
            s0[b] = ((i >> b) & 1) ? 1.0f : -1.0f;
            s1[b] = ((i >> (b + 4)) & 1) ? 1.0f : -1.0f;
        }
        sign_lut[i].v0 = vld1q_f32(s0);
        sign_lut[i].v1 = vld1q_f32(s1);
    }

    for (size_t i = 0; i < M; ++i) {
        const float* rowA = &A[i * K];
        float* rowC = &C[i * K];

        for (size_t j_block = 0; j_block < K_ints; j_block += 2) {
            float32x4_t acc00 = vdupq_n_f32(0.0f), acc01 = vdupq_n_f32(0.0f), acc02 = vdupq_n_f32(0.0f), acc03 = vdupq_n_f32(0.0f),
                         acc04 = vdupq_n_f32(0.0f), acc05 = vdupq_n_f32(0.0f), acc06 = vdupq_n_f32(0.0f), acc07 = vdupq_n_f32(0.0f),
                         acc10 = vdupq_n_f32(0.0f), acc11 = vdupq_n_f32(0.0f), acc12 = vdupq_n_f32(0.0f), acc13 = vdupq_n_f32(0.0f),
                         acc14 = vdupq_n_f32(0.0f), acc15 = vdupq_n_f32(0.0f), acc16 = vdupq_n_f32(0.0f), acc17 = vdupq_n_f32(0.0f);

            for (size_t p = 0; p < K; p += 4) {
                const float32x4_t v_a0 = vdupq_n_f32(rowA[p]);
                const float32x4_t v_a1 = vdupq_n_f32(rowA[p+1]);
                const float32x4_t v_a2 = vdupq_n_f32(rowA[p+2]);
                const float32x4_t v_a3 = vdupq_n_f32(rowA[p+3]);

                const uint32_t p0_0 = B[p * K_ints + j_block];
                const uint32_t p0_1 = B[(p + 1) * K_ints + j_block];
                const uint32_t p0_2 = B[(p + 2) * K_ints + j_block];
                const uint32_t p0_3 = B[(p + 3) * K_ints + j_block];
                
                acc00 = vmlaq_f32(acc00, sign_lut[p0_0 & 0xFF].v0, v_a0);
                acc01 = vmlaq_f32(acc01, sign_lut[p0_0 & 0xFF].v1, v_a0);
                acc02 = vmlaq_f32(acc02, sign_lut[(p0_0 >> 8) & 0xFF].v0, v_a0);
                acc03 = vmlaq_f32(acc03, sign_lut[(p0_0 >> 8) & 0xFF].v1, v_a0);
                acc04 = vmlaq_f32(acc04, sign_lut[(p0_0 >> 16) & 0xFF].v0, v_a0);
                acc05 = vmlaq_f32(acc05, sign_lut[(p0_0 >> 16) & 0xFF].v1, v_a0);
                acc06 = vmlaq_f32(acc06, sign_lut[(p0_0 >> 24) & 0xFF].v0, v_a0);
                acc07 = vmlaq_f32(acc07, sign_lut[(p0_0 >> 24) & 0xFF].v1, v_a0);

                acc00 = vmlaq_f32(acc00, sign_lut[p0_1 & 0xFF].v0, v_a1);
                acc01 = vmlaq_f32(acc01, sign_lut[p0_1 & 0xFF].v1, v_a1);
                acc02 = vmlaq_f32(acc02, sign_lut[(p0_1 >> 8) & 0xFF].v0, v_a1);
                acc03 = vmlaq_f32(acc03, sign_lut[(p0_1 >> 8) & 0xFF].v1, v_a1);
                acc04 = vmlaq_f32(acc04, sign_lut[(p0_1 >> 16) & 0xFF].v0, v_a1);
                acc05 = vmlaq_f32(acc05, sign_lut[(p0_1 >> 16) & 0xFF].v1, v_a1);
                acc06 = vmlaq_f32(acc06, sign_lut[(p0_1 >> 24) & 0xFF].v0, v_a1);
                acc07 = vmlaq_f32(acc07, sign_lut[(p0_1 >> 24) & 0xFF].v1, v_a1);

                acc00 = vmlaq_f32(acc00, sign_lut[p0_2 & 0xFF].v0, v_a2);
                acc01 = vmlaq_f32(acc01, sign_lut[p0_2 & 0xFF].v1, v_a2);
                acc02 = vmlaq_f32(acc02, sign_lut[(p0_2 >> 8) & 0xFF].v0, v_a2);
                acc03 = vmlaq_f32(acc03, sign_lut[(p0_2 >> 8) & 0xFF].v1, v_a2);
                acc04 = vmlaq_f32(acc04, sign_lut[(p0_2 >> 16) & 0xFF].v0, v_a2);
                acc05 = vmlaq_f32(acc05, sign_lut[(p0_2 >> 16) & 0xFF].v1, v_a2);
                acc06 = vmlaq_f32(acc06, sign_lut[(p0_2 >> 24) & 0xFF].v0, v_a2);
                acc07 = vmlaq_f32(acc07, sign_lut[(p0_2 >> 24) & 0xFF].v1, v_a2);

                acc00 = vmlaq_f32(acc00, sign_lut[p0_3 & 0xFF].v0, v_a3);
                acc01 = vmlaq_f32(acc01, sign_lut[p0_3 & 0xFF].v1, v_a3);
                acc02 = vmlaq_f32(acc02, sign_lut[(p0_3 >> 8) & 0xFF].v0, v_a3);
                acc03 = vmlaq_f32(acc03, sign_lut[(p0_3 >> 8) & 0xFF].v1, v_a3);
                acc04 = vmlaq_f32(acc04, sign_lut[(p0_3 >> 16) & 0xFF].v0, v_a3);
                acc05 = vmlaq_f32(acc05, sign_lut[(p0_3 >> 16) & 0xFF].v1, v_a3);
                acc06 = vmlaq_f32(acc06, sign_lut[(p0_3 >> 24) & 0xFF].v0, v_a3);
                acc07 = vmlaq_f32(acc07, sign_lut[(p0_3 >> 24) & 0xFF].v1, v_a3);

                if (j_block + 1 < K_ints) {
                    const uint32_t p1_0 = B[p * K_ints + j_block + 1];
                    const uint32_t p1_1 = B[(p + 1) * K_ints + j_block + 1];
                    const uint32_t p1_2 = B[(p + 2) * K_ints + j_block + 1];
                    const uint32_t p1_3 = B[(p + 3) * K_ints + j_block + 1];

                    acc10 = vmlaq_f32(acc10, sign_lut[p1_0 & 0xFF].v0, v_a0);
                    acc11 = vmlaq_f32(acc11, sign_lut[p1_0 & 0xFF].v1, v_a0);
                    acc12 = vmlaq_f32(acc12, sign_lut[(p1_0 >> 8) & 0xFF].v0, v_a0);
                    acc13 = vmlaq_f32(acc13, sign_lut[(p1_0 >> 8) & 0xFF].v1, v_a0);
                    acc14 = vmlaq_f32(acc14, sign_lut[(p1_0 >> 16) & 0xFF].v0, v_a0);
                    acc15 = vmlaq_f32(acc15, sign_lut[(p1_0 >> 16) & 0xFF].v1, v_a0);
                    acc16 = vmlaq_f32(acc16, sign_lut[(p1_0 >> 24) & 0xFF].v0, v_a0);
                    acc17 = vmlaq_f32(acc17, sign_lut[(p1_0 >> 24) & 0xFF].v1, v_a0);

                    acc10 = vmlaq_f32(acc10, sign_lut[p1_1 & 0xFF].v0, v_a1);
                    acc11 = vmlaq_f32(acc11, sign_lut[p1_1 & 0xFF].v1, v_a1);
                    acc12 = vmlaq_f32(acc12, sign_lut[(p1_1 >> 8) & 0xFF].v0, v_a1);
                    acc13 = vmlaq_f32(acc13, sign_lut[(p1_1 >> 8) & 0xFF].v1, v_a1);
                    acc14 = vmlaq_f32(acc14, sign_lut[(p1_1 >> 16) & 0xFF].v0, v_a1);
                    acc15 = vmlaq_f32(acc15, sign_lut[(p1_1 >> 16) & 0xFF].v1, v_a1);
                    acc16 = vmlaq_f32(acc16, sign_lut[(p1_1 >> 24) & 0xFF].v0, v_a1);
                    acc17 = vmlaq_f32(acc17, sign_lut[(p1_1 >> 24) & 0xFF].v1, v_a1);

                    acc10 = vmlaq_f32(acc10, sign_lut[p1_2 & 0xFF].v0, v_a2);
                    acc11 = vmlaq_f32(acc11, sign_lut[p1_2 & 0xFF].v1, v_a2);
                    acc12 = vmlaq_f32(acc12, sign_lut[(p1_2 >> 8) & 0xFF].v0, v_a2);
                    acc13 = vmlaq_f32(acc13, sign_lut[(p1_2 >> 8) & 0xFF].v1, v_a2);
                    acc14 = vmlaq_f32(acc14, sign_lut[(p1_2 >> 16) & 0xFF].v0, v_a2);
                    acc15 = vmlaq_f32(acc15, sign_lut[(p1_2 >> 16) & 0xFF].v1, v_a2);
                    acc16 = vmlaq_f32(acc16, sign_lut[(p1_2 >> 24) & 0xFF].v0, v_a2);
                    acc17 = vmlaq_f32(acc17, sign_lut[(p1_2 >> 24) & 0xFF].v1, v_a2);

                    acc10 = vmlaq_f32(acc10, sign_lut[p1_3 & 0xFF].v0, v_a3);
                    acc11 = vmlaq_f32(acc11, sign_lut[p1_3 & 0xFF].v1, v_a3);
                    acc12 = vmlaq_f32(acc12, sign_lut[(p1_3 >> 8) & 0xFF].v0, v_a3);
                    acc13 = vmlaq_f32(acc13, sign_lut[(p1_3 >> 8) & 0xFF].v1, v_a3);
                    acc14 = vmlaq_f32(acc14, sign_lut[(p1_3 >> 16) & 0xFF].v0, v_a3);
                    acc15 = vmlaq_f32(acc15, sign_lut[(p1_3 >> 16) & 0xFF].v1, v_a3);
                    acc16 = vmlaq_f32(acc16, sign_lut[(p1_3 >> 24) & 0xFF].v0, v_a3);
                    acc17 = vmlaq_f32(acc17, sign_lut[(p1_3 >> 24) & 0xFF].v1, v_a3);
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
