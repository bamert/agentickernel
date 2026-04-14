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

    for (size_t i = 0; i < M; i += 4) {
        const float* rowA0 = &A[i * K];
        const float* rowA1 = (i + 1 < M) ? &A[(i + 1) * K] : nullptr;
        const float* rowA2 = (i + 2 < M) ? &A[(i + 2) * K] : nullptr;
        const float* rowA3 = (i + 3 < M) ? &A[(i + 3) * K] : nullptr;
        
        for (size_t j_block = 0; j_block < K_ints; ++j_block) {
            float32x4_t acc0[8], acc1[8], acc2[8], acc3[8];
            for(int n=0; n<8; ++n) {
                acc0[n] = vdupq_n_f32(0.0f);
                acc1[n] = vdupq_n_f32(0.0f);
                acc2[n] = vdupq_n_f32(0.0f);
                acc3[n] = vdupq_n_f32(0.0f);
            }

            for (size_t p = 0; p < K; p += 4) {
                const uint32_t p0 = B[p * K_ints + j_block];
                const uint32_t p1 = B[(p + 1) * K_ints + j_block];
                const uint32_t p2 = B[(p + 2) * K_ints + j_block];
                const uint32_t p3 = B[(p + 3) * K_ints + j_block];

                const float32x4_t va0_0 = vdupq_n_f32(rowA0[p]);
                const float32x4_t va0_1 = vdupq_n_f32(rowA0[p+1]);
                const float32x4_t va0_2 = vdupq_n_f32(rowA0[p+2]);
                const float32x4_t va0_3 = vdupq_n_f32(rowA0[p+3]);

                // Row 0 updates
                acc0[0] = vmlaq_f32(acc0[0], sign_lut[p0 & 0xFF].v0, va0_0);
                acc0[1] = vmlaq_f32(acc0[1], sign_lut[p0 & 0xFF].v1, va0_0);
                acc0[2] = vmlaq_f32(acc0[2], sign_lut[(p0 >> 8) & 0xFF].v0, va0_0);
                acc0[3] = vmlaq_f32(acc0[3], sign_lut[(p0 >> 8) & 0xFF].v1, va0_0);
                acc0[4] = vmlaq_f32(acc0[4], sign_lut[(p0 >> 16) & 0xFF].v0, va0_0);
                acc0[5] = vmlaq_f32(acc0[5], sign_lut[(p0 >> 16) & 0xFF].v1, va0_0);
                acc0[6] = vmlaq_f32(acc0[6], sign_lut[(p0 >> 24) & 0xFF].v0, va0_0);
                acc0[7] = vmlaq_f32(acc0[7], sign_lut[(p0 >> 24) & 0xFF].v1, va0_0);

                acc0[0] = vmlaq_f32(acc0[0], sign_lut[p1 & 0xFF].v0, va0_1);
                acc0[1] = vmlaq_f32(acc0[1], sign_lut[p1 & 0xFF].v1, va0_1);
                acc0[2] = vmlaq_f32(acc0[2], sign_lut[(p1 >> 8) & 0xFF].v0, va0_1);
                acc0[3] = vmlaq_f32(acc0[3], sign_lut[(p1 >> 8) & 0xFF].v1, va0_1);
                acc0[4] = vmlaq_f32(acc0[4], sign_lut[(p1 >> 16) & 0xFF].v0, va0_1);
                acc0[5] = vmlaq_f32(acc0[5], sign_lut[(p1 >> 16) & 0xFF].v1, va0_1);
                acc0[6] = vmlaq_f32(acc0[6], sign_lut[(p1 >> 24) & 0xFF].v0, va0_1);
                acc0[7] = vmlaq_f32(acc0[7], sign_lut[(p1 >> 24) & 0xFF].v1, va0_1);

                acc0[0] = vmlaq_f32(acc0[0], sign_lut[p2 & 0xFF].v0, va0_2);
                acc0[1] = vmlaq_f32(acc0[1], sign_lut[p2 & 0xFF].v1, va0_2);
                acc0[2] = vmlaq_f32(acc0[2], sign_lut[(p2 >> 8) & 0xFF].v0, va0_2);
                acc0[3] = vmlaq_f32(acc0[3], sign_lut[(p2 >> 8) & 0xFF].v1, va0_2);
                acc0[4] = vmlaq_f32(acc0[4], sign_lut[(p2 >> 16) & 0xFF].v0, va0_2);
                acc0[5] = vmlaq_f32(acc0[5], sign_lut[(p2 >> 16) & 0xFF].v1, va0_2);
                acc0[6] = vmlaq_f32(acc0[6], sign_lut[(p2 >> 24) & 0xFF].v0, va0_2);
                acc0[7] = vmlaq_f32(acc0[7], sign_lut[(p2 >> 24) & 0xFF].v1, va0_2);

                acc0[0] = vmlaq_f32(acc0[0], sign_lut[p3 & 0xFF].v0, va0_3);
                acc0[1] = vmlaq_f32(acc0[1], sign_lut[p3 & 0xFF].v1, va0_3);
                acc0[2] = vmlaq_f32(acc0[2], sign_lut[(p3 >> 8) & 0xFF].v0, va0_3);
                acc0[3] = vmlaq_f32(acc0[3], sign_lut[(p3 >> 8) & 0xFF].v1, va0_3);
                acc0[4] = vmlaq_f32(acc0[4], sign_lut[(p3 >> 16) & 0xFF].v0, va0_3);
                acc0[5] = vmlaq_f32(acc0[5], sign_lut[(p3 >> 16) & 0xFF].v1, va0_3);
                acc0[6] = vmlaq_f32(acc0[6], sign_lut[(p3 >> 24) & 0xFF].v0, va0_3);
                acc0[7] = vmlaq_f32(acc0[7], sign_lut[(p3 >> 24) & 0xFF].v1, va0_3);

                if (rowA1) {
                    const float32x4_t va1_0 = vdupq_n_f32(rowA1[p]);
                    const float32x4_t va1_1 = vdupq_n_f32(rowA1[p+1]);
                    const float32x4_t va1_2 = vdupq_n_f32(rowA1[p+2]);
                    const float32x4_t va1_3 = vdupq_n_f32(rowA1[p+3]);
                    acc1[0] = vmlaq_f32(acc1[0], sign_lut[p0 & 0xFF].v0, va1_0);
                    acc1[1] = vmlaq_f32(acc1[1], sign_lut[p0 & 0xFF].v1, va1_0);
                    acc1[2] = vmlaq_f32(acc1[2], sign_lut[(p0 >> 8) & 0xFF].v0, va1_0);
                    acc1[3] = vmlaq_f32(acc1[3], sign_lut[(p0 >> 8) & 0xFF].v1, va1_0);
                    acc1[4] = vmlaq_f32(acc1[4], sign_lut[(p0 >> 16) & 0xFF].v0, va1_0);
                    acc1[5] = vmlaq_f32(acc1[5], sign_lut[(p0 >> 16) & 0xFF].v1, va1_0);
                    acc1[6] = vmlaq_f32(acc1[6], sign_lut[(p0 >> 24) & 0xFF].v0, va1_0);
                    acc1[7] = vmlaq_f32(acc1[7], sign_lut[(p0 >> 24) & 0xFF].v1, va1_0);

                    acc1[0] = vmlaq_f32(acc1[0], sign_lut[p1 & 0xFF].v0, va1_1);
                    acc1[1] = vmlaq_f32(acc1[1], sign_lut[p1 & 0xFF].v1, va1_1);
                    acc1[2] = vmlaq_f32(acc1[2], sign_lut[(p1 >> 8) & 0xFF].v0, va1_1);
                    acc1[3] = vmlaq_f32(acc1[3], sign_lut[(p1 >> 8) & 0xFF].v1, va1_1);
                    acc1[4] = vmlaq_f32(acc1[4], sign_lut[(p1 >> 16) & 0xFF].v0, va1_1);
                    acc1[5] = vmlaq_f32(acc1[5], sign_lut[(p1 >> 16) & 0xFF].v1, va1_1);
                    acc1[6] = vmlaq_f32(acc1[6], sign_lut[(p1 >> 24) & 0xFF].v0, va1_1);
                    acc1[7] = vmlaq_f32(acc1[7], sign_lut[(p1 >> 24) & 0xFF].v1, va1_1);

                    acc1[0] = vmlaq_f32(acc1[0], sign_lut[p2 & 0xFF].v0, va1_2);
                    acc1[1] = vmlaq_f32(acc1[1], sign_lut[p2 & 0xFF].v1, va1_2);
                    acc1[2] = vmlaq_f32(acc1[2], sign_lut[(p2 >> 8) & 0xFF].v0, va1_2);
                    acc1[3] = vmlaq_f32(acc1[3], sign_lut[(p2 >> 8) & 0xFF].v1, va1_2);
                    acc1[4] = vmlaq_f32(acc1[4], sign_lut[(p2 >> 16) & 0xFF].v0, va1_2);
                    acc1[5] = vmlaq_f32(acc1[5], sign_lut[(p2 >> 16) & 0xFF].v1, va1_2);
                    acc1[6] = vmlaq_f32(acc1[6], sign_lut[(p2 >> 24) & 0xFF].v0, va1_2);
                    acc1[7] = vmlaq_f32(acc1[7], sign_lut[(p2 >> 24) & 0xFF].v1, va1_2);

                    acc1[0] = vmlaq_f32(acc1[0], sign_lut[p3 & 0xFF].v0, va1_3);
                    acc1[1] = vmlaq_f32(acc1[1], sign_lut[p3 & 0xFF].v1, va1_3);
                    acc1[2] = vmlaq_f32(acc1[2], sign_lut[(p3 >> 8) & 0xFF].v0, va1_3);
                    acc1[3] = vmlaq_f32(acc1[3], sign_lut[(p3 >> 8) & 0xFF].v1, va1_3);
                    acc1[4] = vmlaq_f32(acc1[4], sign_lut[(p3 >> 16) & 0xFF].v0, va1_3);
                    acc1[5] = vmlaq_f32(acc1[5], sign_lut[(p3 >> 16) & 0xFF].v1, va1_3);
                    acc1[6] = vmlaq_f32(acc1[6], sign_lut[(p3 >> 24) & 0xFF].v0, va1_3);
                    acc1[7] = vmlaq_f32(acc1[7], sign_lut[(p3 >> 24) & 0xFF].v1, va1_3);
                }
                if (rowA2) {
                    const float32x4_t va2_0 = vdupq_n_f32(rowA2[p]);
                    const float32x4_t va2_1 = vdupq_n_f32(rowA2[p+1]);
                    const float32x4_t va2_2 = vdupq_n_f32(rowA2[p+2]);
                    const float32x4_t va2_3 = vdupq_n_f32(rowA2[p+3]);
                    acc2[0] = vmlaq_f32(acc2[0], sign_lut[p0 & 0xFF].v0, va2_0);
                    acc2[1] = vmlaq_f32(acc2[1], sign_lut[p0 & 0xFF].v1, va2_0);
                    acc2[2] = vmlaq_f32(acc2[2], sign_lut[(p0 >> 8) & 0xFF].v0, va2_0);
                    acc2[3] = vmlaq_f32(acc2[3], sign_lut[(p0 >> 8) & 0xFF].v1, va2_0);
                    acc2[4] = vmlaq_f32(acc2[4], sign_lut[(p0 >> 16) & 0xFF].v0, va2_0);
                    acc2[5] = vmlaq_f32(acc2[5], sign_lut[(p0 >> 16) & 0xFF].v1, va2_0);
                    acc2[6] = vmlaq_f32(acc2[6], sign_lut[(p0 >> 24) & 0xFF].v0, va2_0);
                    acc2[7] = vmlaq_f32(acc2[7], sign_lut[(p0 >> 24) & 0xFF].v1, va2_0);

                    acc2[0] = vmlaq_f32(acc2[0], sign_lut[p1 & 0xFF].v0, va2_1);
                    acc2[1] = vmlaq_f32(acc2[1], sign_lut[p1 & 0xFF].v1, va2_1);
                    acc2[2] = vmlaq_f32(acc2[2], sign_lut[(p1 >> 8) & 0xFF].v0, va2_1);
                    acc2[3] = vmlaq_f32(acc2[3], sign_lut[(p1 >> 8) & 0xFF].v1, va2_1);
                    acc2[4] = vmlaq_f32(acc2[4], sign_lut[(p1 >> 16) & 0xFF].v0, va2_1);
                    acc2[5] = vmlaq_f32(acc2[5], sign_lut[(p1 >> 16) & 0xFF].v1, va2_1);
                    acc2[6] = vmlaq_f32(acc2[6], sign_lut[(p1 >> 24) & 0xFF].v0, va2_1);
                    acc2[7] = vmlaq_f32(acc2[7], sign_lut[(p1 >> 24) & 0xFF].v1, va2_1);

                    acc2[0] = vmlaq_f32(acc2[0], sign_lut[p2 & 0xFF].v0, va2_2);
                    acc2[1] = vmlaq_f32(acc2[1], sign_lut[p2 & 0xFF].v1, va2_2);
                    acc2[2] = vmlaq_f32(acc2[2], sign_lut[(p2 >> 8) & 0xFF].v0, va2_2);
                    acc2[3] = vmlaq_f32(acc2[3], sign_lut[(p2 >> 8) & 0xFF].v1, va2_2);
                    acc2[4] = vmlaq_f32(acc2[4], sign_lut[(p2 >> 16) & 0xFF].v0, va2_2);
                    acc2[5] = vmlaq_f32(acc2[5], sign_lut[(p2 >> 16) & 0xFF].v1, va2_2);
                    acc2[6] = vmlaq_f32(acc2[6], sign_lut[(p2 >> 24) & 0xFF].v0, va2_2);
                    acc2[7] = vmlaq_f32(acc2[7], sign_lut[(p2 >> 24) & 0xFF].v1, va2_2);

                    acc2[0] = vmlaq_f32(acc2[0], sign_lut[p3 & 0xFF].v0, va2_3);
                    acc2[1] = vmlaq_f32(acc2[1], sign_lut[p3 & 0xFF].v1, va2_3);
                    acc2[2] = vmlaq_f32(acc2[2], sign_lut[(p3 >> 8) & 0xFF].v0, va2_3);
                    acc2[3] = vmlaq_f32(acc2[3], sign_lut[(p3 >> 8) & 0xFF].v1, va2_3);
                    acc2[4] = vmlaq_f32(acc2[4], sign_lut[(p3 >> 16) & 0xFF].v0, va2_3);
                    acc2[5] = vmlaq_f32(acc2[5], sign_lut[(p3 >> 16) & 0xFF].v1, va2_3);
                    acc2[6] = vmlaq_f32(acc2[6], sign_lut[(p3 >> 24) & 0xFF].v0, va2_3);
                    acc2[7] = vmlaq_f32(acc2[7], sign_lut[(p3 >> 24) & 0xFF].v1, va2_3);
                }
                if (rowA3) {
                    const float32x4_t va3_0 = vdupq_n_f32(rowA3[p]);
                    const float32x4_t va3_1 = vdupq_n_f32(rowA3[p+1]);
                    const float32x4_t va3_2 = vdupq_n_f32(rowA3[p+2]);
                    const float32x4_t va3_3 = vdupq_n_f32(rowA3[p+3]);
                    acc3[0] = vmlaq_f32(acc3[0], sign_lut[p0 & 0xFF].v0, va3_0);
                    acc3[1] = vmlaq_f32(acc3[1], sign_lut[p0 & 0xFF].v1, va3_0);
                    acc3[2] = vmlaq_f32(acc3[2], sign_lut[(p0 >> 8) & 0xFF].v0, va3_0);
                    acc3[3] = vmlaq_f32(acc3[3], sign_lut[(p0 >> 8) & 0xFF].v1, va3_0);
                    acc3[4] = vmlaq_f32(acc3[4], sign_lut[(p0 >> 16) & 0xFF].v0, va3_0);
                    acc3[5] = vmlaq_f32(acc3[5], sign_lut[(p0 >> 16) & 0xFF].v1, va3_0);
                    acc3[6] = vmlaq_f32(acc3[6], sign_lut[(p0 >> 24) & 0xFF].v0, va3_0);
                    acc3[7] = vmlaq_f32(acc3[7], sign_lut[(p0 >> 24) & 0xFF].v1, va3_0);

                    acc3[0] = vmlaq_f32(acc3[0], sign_lut[p1 & 0xFF].v0, va3_1);
                    acc3[1] = vmlaq_f32(acc3[1], sign_lut[p1 & 0xFF].v1, va3_1);
                    acc3[2] = vmlaq_f32(acc3[2], sign_lut[(p1 >> 8) & 0xFF].v0, va3_1);
                    acc3[3] = vmlaq_f32(acc3[3], sign_lut[(p1 >> 8) & 0xFF].v1, va3_1);
                    acc3[4] = vmlaq_f32(acc3[4], sign_lut[(p1 >> 16) & 0xFF].v0, va3_1);
                    acc3[5] = vmlaq_f32(acc3[5], sign_lut[(p1 >> 16) & 0xFF].v1, va3_1);
                    acc3[6] = vmlaq_f32(acc3[6], sign_lut[(p1 >> 24) & 0xFF].v0, va3_1);
                    acc3[7] = vmlaq_f32(acc3[7], sign_lut[(p1 >> 24) & 0xFF].v1, va3_1);

                    acc3[0] = vmlaq_f32(acc3[0], sign_lut[p2 & 0xFF].v0, va3_2);
                    acc3[1] = vmlaq_f32(acc3[1], sign_lut[p2 & 0xFF].v1, va3_2);
                    acc3[2] = vmlaq_f32(acc3[2], sign_lut[(p2 >> 8) & 0xFF].v0, va3_2);
                    acc3[3] = vmlaq_f32(acc3[3], sign_lut[(p2 >> 8) & 0xFF].v1, va3_2);
                    acc3[4] = vmlaq_f32(acc3[4], sign_lut[(p2 >> 16) & 0xFF].v0, va3_2);
                    acc3[5] = vmlaq_f32(acc3[5], sign_lut[(p2 >> 16) & 0xFF].v1, va3_2);
                    acc3[6] = vmlaq_f32(acc3[6], sign_lut[(p2 >> 24) & 0xFF].v0, va3_2);
                    acc3[7] = vmlaq_f32(acc3[7], sign_lut[(p2 >> 24) & 0xFF].v1, va3_2);

                    acc3[0] = vmlaq_f32(acc3[0], sign_lut[p3 & 0xFF].v0, va3_3);
                    acc3[1] = vmlaq_f32(acc3[1], sign_lut[p3 & 0xFF].v1, va3_3);
                    acc3[2] = vmlaq_f32(acc3[2], sign_lut[(p3 >> 8) & 0xFF].v0, va3_3);
                    acc3[3] = vmlaq_f32(acc3[3], sign_lut[(p3 >> 8) & 0xFF].v1, va3_3);
                    acc3[4] = vmlaq_f32(acc3[4], sign_lut[(p3 >> 16) & 0xFF].v0, va3_3);
                    acc3[5] = vmlaq_f32(acc3[5], sign_lut[(p3 >> 16) & 0xFF].v1, va3_3);
                    acc3[6] = vmlaq_f32(acc3[6], sign_lut[(p3 >> 24) & 0xFF].v0, va3_3);
                    acc3[7] = vmlaq_f32(acc3[7], sign_lut[(p3 >> 24) & 0xFF].v1, va3_3);
                }
            }

            float* out_ptr0 = &C[i * K + j_block * 32];
            for(int n=0; n<8; ++n) vst1q_f32(out_ptr0 + n*4, acc0[n]);
            if (i + 1 < M) {
                float* out_ptr1 = &C[(i + 1) * K + j_block * 32];
                for(int n=0; n<8; ++n) vst1q_f32(out_ptr1 + n*4, acc1[n]);
            }
            if (i + 2 < M) {
                float* out_ptr2 = &C[(i + 2) * K + j_block * 32];
                for(int n=0; n<8; ++n) vst1q_f32(out_ptr2 + n*4, acc2[n]);
            }
            if (i + 3 < M) {
                float* out_ptr3 = &C[(i + 3) * K + j_block * 32];
                for(int n=0; n<8; ++n) vst1q_f32(out_ptr3 + n*4, acc3[n]);
            }
        }
    }
}
