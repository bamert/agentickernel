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

    float32x4_t lut0[256];
    float32x4_t lut1[256];

    for (int i = 0; i < 256; ++i) {
        float s0[4], s1[4];
        for (int b = 0; b < 4; ++b) {
            s0[b] = ((i >> b) & 1) ? 1.0f : -1.0f;
            s1[b] = ((i >> (b + 4)) & 1) ? 1.0f : -1.0f;
        }
        lut0[i] = vld1q_f32(s0);
        lut1[i] = vld1q_f32(s1);
    }

    size_t i = 0;
    while (i + 1 < M) {
        const float* rowA0 = &A[i * K];
        const float* rowA1 = &A[(i + 1) * K];
        
        for (size_t j_block = 0; j_block < K_ints; ++j_block) {
            float32x4_t acc0[8], acc1[8];
            for (int n = 0; n < 8; ++n) {
                acc0[n] = vdupq_n_f32(0.0f);
                acc1[n] = vdupq_n_f32(0.0f);
            }

            for (size_t p = 0; p < K; p += 4) {
                const uint32_t b0 = B[p * K_ints + j_block];
                const uint32_t b1 = B[(p + 1) * K_ints + j_block];
                const uint32_t b2 = B[(p + 2) * K_ints + j_block];
                const uint32_t b3 = B[(p + 3) * K_ints + j_block];

                const float32x4_t va0_0 = vdupq_n_f32(rowA0[p]);
                const float32x4_t va0_1 = vdupq_n_f32(rowA0[p+1]);
                const float32x4_t va0_2 = vdupq_n_f32(rowA0[p+2]);
                const float32x4_t va0_3 = vdupq_n_f32(rowA0[p+3]);
                
                const float32x4_t va1_0 = vdupq_n_f32(rowA1[p]);
                const float32x4_t va1_1 = vdupq_n_f32(rowA1[p+1]);
                const float32x4_t va1_2 = vdupq_n_f32(rowA1[p+2]);
                const float32x4_t va1_3 = vdupq_n_f32(rowA1[p+3]);

                // B0
                const uint8_t b0_0 = b0 & 0xFF;
                const uint8_t b0_1 = (b0 >> 8) & 0xFF;
                const uint8_t b0_2 = (b0 >> 16) & 0xFF;
                const uint8_t b0_3 = (b0 >> 24) & 0xFF;

                acc0[0] = vmlaq_f32(acc0[0], lut0[b0_0], va0_0);
                acc0[1] = vmlaq_f32(acc0[1], lut1[b0_0], va0_0);
                acc0[2] = vmlaq_f32(acc0[2], lut0[b0_1], va0_0);
                acc0[3] = vmlaq_f32(acc0[3], lut1[b0_1], va0_0);
                acc0[4] = vmlaq_f32(acc0[4], lut0[b0_2], va0_0);
                acc0[5] = vmlaq_f32(acc0[5], lut1[b0_2], va0_0);
                acc0[6] = vmlaq_f32(acc0[6], lut0[b0_3], va0_0);
                acc0[7] = vmlaq_f32(acc0[7], lut1[b0_3], va0_0);
                acc1[0] = vmlaq_f32(acc1[0], lut0[b0_0], va1_0);
                acc1[1] = vmlaq_f32(acc1[1], lut1[b0_0], va1_0);
                acc1[2] = vmlaq_f32(acc1[2], lut0[b0_1], va1_0);
                acc1[3] = vmlaq_f32(acc1[3], lut1[b0_1], va1_0);
                acc1[4] = vmlaq_f32(acc1[4], lut0[b0_2], va1_0);
                acc1[5] = vmlaq_f32(acc1[5], lut1[b0_2], va1_0);
                acc1[6] = vmlaq_f32(acc1[6], lut0[b0_3], va1_0);
                acc1[7] = vmlaq_f32(acc1[7], lut1[b0_3], va1_0);

                // B1
                const uint8_t b1_0 = b1 & 0xFF;
                const uint8_t b1_1 = (b1 >> 8) & 0xFF;
                const uint8_t b1_2 = (b1 >> 16) & 0xFF;
                const uint8_t b1_3 = (b1 >> 24) & 0xFF;
                acc0[0] = vmlaq_f32(acc0[0], lut0[b1_0], va0_1);
                acc0[1] = vmlaq_f32(acc0[1], lut1[b1_0], va0_1);
                acc0[2] = vmlaq_f32(acc0[2], lut0[b1_1], va0_1);
                acc0[3] = vmlaq_f32(acc0[3], lut1[b1_1], va0_1);
                acc0[4] = vmlaq_f32(acc0[4], lut0[b1_2], va0_1);
                acc0[5] = vmlaq_f32(acc0[5], lut1[b1_2], va0_1);
                acc0[6] = vmlaq_f32(acc0[6], lut0[b1_3], va0_1);
                acc0[7] = vmlaq_f32(acc0[7], lut1[b1_3], va0_1);
                acc1[0] = vmlaq_f32(acc1[0], lut0[b1_0], va1_1);
                acc1[1] = vmlaq_f32(acc1[1], lut1[b1_0], va1_1);
                acc1[2] = vmlaq_f32(acc1[2], lut0[b1_1], va1_1);
                acc1[3] = vmlaq_f32(acc1[3], lut1[b1_1], va1_1);
                acc1[4] = vmlaq_f32(acc1[4], lut0[b1_2], va1_1);
                acc1[5] = vmlaq_f32(acc1[5], lut1[b1_2], va1_1);
                acc1[6] = vmlaq_f32(acc1[6], lut0[b1_3], va1_1);
                acc1[7] = vmlaq_f32(acc1[7], lut1[b1_3], va1_1);

                // B2
                const uint8_t b2_0 = b2 & 0xFF;
                const uint8_t b2_1 = (b2 >> 8) & 0xFF;
                const uint8_t b2_2 = (b2 >> 16) & 0xFF;
                const uint8_t b2_3 = (b2 >> 24) & 0xFF;
                acc0[0] = vmlaq_f32(acc0[0], lut0[b2_0], va0_2);
                acc0[1] = vmlaq_f32(acc0[1], lut1[b2_0], va0_2);
                acc0[2] = vmlaq_f32(acc0[2], lut0[b2_1], va0_2);
                acc0[3] = vmlaq_f32(acc0[3], lut1[b2_1], va0_2);
                acc0[4] = vmlaq_f32(acc0[4], lut0[b2_2], va0_2);
                acc0[5] = vmlaq_f32(acc0[5], lut1[b2_2], va0_2);
                acc0[6] = vmlaq_f32(acc0[6], lut0[b2_3], va0_2);
                acc0[7] = vmlaq_f32(acc0[7], lut1[b2_3], va0_2);
                acc1[0] = vmlaq_f32(acc1[0], lut0[b2_0], va1_2);
                acc1[1] = vmlaq_f32(acc1[1], lut1[b2_0], va1_2);
                acc1[2] = vmlaq_f32(acc1[2], lut0[b2_1], va1_2);
                acc1[3] = vmlaq_f32(acc1[3], lut1[b2_1], va1_2);
                acc1[4] = vmlaq_f32(acc1[4], lut0[b2_2], va1_2);
                acc1[5] = vmlaq_f32(acc1[5], lut1[b2_2], va1_2);
                acc1[6] = vmlaq_f32(acc1[6], lut0[b2_3], va1_2);
                acc1[7] = vmlaq_f32(acc1[7], lut1[b2_3], va1_2);

                // B3
                const uint8_t b3_0 = b3 & 0xFF;
                const uint8_t b3_1 = (b3 >> 8) & 0xFF;
                const uint8_t b3_2 = (b3 >> 16) & 0xFF;
                const uint8_t b3_3 = (b3 >> 24) & 0xFF;
                acc0[0] = vmlaq_f32(acc0[0], lut0[b3_0], va0_3);
                acc0[1] = vmlaq_f32(acc0[1], lut1[b3_0], va0_3);
                acc0[2] = vmlaq_f32(acc0[2], lut0[b3_1], va0_3);
                acc0[3] = vmlaq_f32(acc0[3], lut1[b3_1], va0_3);
                acc0[4] = vmlaq_f32(acc0[4], lut0[b3_2], va0_3);
                acc0[5] = vmlaq_f32(acc0[5], lut1[b3_2], va0_3);
                acc0[6] = vmlaq_f32(acc0[6], lut0[b3_3], va0_3);
                acc0[7] = vmlaq_f32(acc0[7], lut1[b3_3], va0_3);
                acc1[0] = vmlaq_f32(acc1[0], lut0[b3_0], va1_3);
                acc1[1] = vmlaq_f32(acc1[1], lut1[b3_0], va1_3);
                acc1[2] = vmlaq_f32(acc1[2], lut0[b3_1], va1_3);
                acc1[3] = vmlaq_f32(acc1[3], lut1[b3_1], va1_3);
                acc1[4] = vmlaq_f32(acc1[4], lut0[b3_2], va1_3);
                acc1[5] = vmlaq_f32(acc1[5], lut1[b3_2], va1_3);
                acc1[6] = vmlaq_f32(acc1[6], lut0[b3_3], va1_3);
                acc1[7] = vmlaq_f32(acc1[7], lut1[b3_3], va1_3);
            }

            float* out0 = &C[i * K + j_block * 32];
            for (int n = 0; n < 8; ++n) vst1q_f32(out0 + n * 4, acc0[n]);
            float* out1 = &C[(i + 1) * K + j_block * 32];
            for (int n = 0; n < 8; ++n) vst1q_f32(out1 + n * 4, acc1[n]);
        }
        i += 2;
    }

    if (i < M) {
        const float* rowA = &A[i * K];
        for (size_t j_block = 0; j_block < K_ints; ++j_block) {
            float32x4_t acc[8];
            for (int n = 0; n < 8; ++n) acc[n] = vdupq_n_f32(0.0f);
            for (size_t p = 0; p < K; ++p) {
                const float32x4_t va = vdupq_n_f32(rowA[p]);
                const uint32_t b = B[p * K_ints + j_block];
                acc[0] = vmlaq_f32(acc[0], lut0[b & 0xFF], va);
                acc[1] = vmlaq_f32(acc[1], lut1[b & 0xFF], va);
                acc[2] = vmlaq_f32(acc[2], lut0[(b >> 8) & 0xFF], va);
                acc[3] = vmlaq_f32(acc[3], lut1[(b >> 8) & 0xFF], va);
                acc[4] = vmlaq_f32(acc[4], lut0[(b >> 16) & 0xFF], va);
                acc[5] = vmlaq_f32(acc[5], lut1[(b >> 16) & 0xFF], va);
                acc[6] = vmlaq_f32(acc[6], lut0[(b >> 24) & 0xFF], va);
                acc[7] = vmlaq_f32(acc[7], lut1[(b >> 24) & 0xFF], va);
            }
            float* out = &C[i * K + j_block * 32];
            for (int n = 0; n < 8; ++n) vst1q_f32(out + n * 4, acc[n]);
        }
    }
}
