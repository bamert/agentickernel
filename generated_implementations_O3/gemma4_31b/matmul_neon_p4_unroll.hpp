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
                         acc04 = vdupq_n_f32(0.0f), acc05 = vdupq_n_f32(0.0f), acc06 = vdupq_n_f32(0.0f), acc07 = vdupq_n_f32(0.0f);
            float32x4_t acc10 = vdupq_n_f32(0.0f), acc11 = vdupq_n_f32(0.0f), acc12 = vdupq_n_f32(0.0f), acc13 = vdupq_n_f32(0.0f),
                         acc14 = vdupq_n_f32(0.0f), acc15 = vdupq_n_f32(0.0f), acc16 = vdupq_n_f32(0.0f), acc17 = vdupq_n_f32(0.0f);

            for (size_t p = 0; p < K; p += 4) {
                float a_vals[4];
                a_vals[0] = rowA[p];
                a_vals[1] = rowA[p + 1];
                a_vals[2] = rowA[p + 2];
                a_vals[3] = rowA[p + 3];

                float32x4_t v_a[4];
                v_a[0] = vdupq_n_f32(a_vals[0]);
                v_a[1] = vdupq_n_f32(a_vals[1]);
                v_a[2] = vdupq_n_f32(a_vals[2]);
                v_a[3] = vdupq_n_f32(a_vals[3]);

                const uint32_t p0[4], p1[4];
                for(int n=0; n<4; ++n) {
                    p0[n] = B[(p + n) * K_ints + j_block];
                    if (j_block + 1 < K_ints) {
                        p1[n] = B[(p + n) * K_ints + j_block + 1];
                    } else {
                        p1[n] = 0;
                    }
                }

                for(int n=0; n<4; ++n) {
                    const uint32_t val0 = p0[n];
                    acc00 = vmlaq_f32(acc00, sign_lut[val0 & 0xFF].v0, v_a[n]);
                    acc01 = vmlaq_f32(acc01, sign_lut[val0 & 0xFF].v1, v_a[n]);
                    acc02 = vmlaq_f32(acc02, sign_lut[(val0 >> 8) & 0xFF].v0, v_a[n]);
                    acc03 = vmlaq_f32(acc03, sign_lut[(val0 >> 8) & 0xFF].v1, v_a[n]);
                    acc04 = vmlaq_f32(acc04, sign_lut[(val0 >> 16) & 0xFF].v0, v_a[n]);
                    acc05 = vmlaq_f32(acc05, sign_lut[(val0 >> 16) & 0xFF].v1, v_a[n]);
                    acc06 = vmlaq_f32(acc06, sign_lut[(val0 >> 24) & 0xFF].v0, v_a[n]);
                    acc07 = vmlaq_f32(acc07, sign_lut[(val0 >> 24) & 0xFF].v1, v_a[n]);

                    if (j_block + 1 < K_ints) {
                        const uint32_t val1 = p1[n];
                        acc10 = vmlaq_f32(acc10, sign_lut[val1 & 0xFF].v0, v_a[n]);
                        acc11 = vmlaq_f32(acc11, sign_lut[val1 & 0xFF].v1, v_a[n]);
                        acc12 = vmlaq_f32(acc12, sign_lut[(val1 >> 8) & 0xFF].v0, v_a[n]);
                        acc13 = vmlaq_f32(acc13, sign_lut[(val1 >> 8) & 0xFF].v1, v_a[n]);
                        acc14 = vmlaq_f32(acc14, sign_lut[(val1 >> 16) & 0xFF].v0, v_a[n]);
                        acc15 = vmlaq_f32(acc15, sign_lut[(val1 >> 16) & 0xFF].v1, v_a[n]);
                        acc16 = vmlaq_f32(acc16, sign_lut[(val1 >> 24) & 0xFF].v0, v_a[n]);
                        acc17 = vmlaq_f32(acc17, sign_lut[(val1 >> 24) & 0xFF].v1, v_a[n]);
                    }
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
