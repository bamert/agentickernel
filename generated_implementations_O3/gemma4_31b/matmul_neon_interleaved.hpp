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

    for (size_t i = 0; i < M; i += 2) {
        const float* rowA0 = &A[i * K];
        const float* rowA1 = (i + 1 < M) ? &A[(i + 1) * K] : nullptr;
        
        for (size_t j_block = 0; j_block < K_ints; ++j_block) {
            float32x4_t acc0[8], acc1[8];
            for (int n = 0; n < 8; ++n) {
                acc0[n] = vdupq_n_f32(0.0f);
                acc1[n] = vdupq_n_f32(0.0f);
            }

            for (size_t p = 0; p < K; p += 2) {
                // Load binary values and float values
                const uint32_t b0 = B[p * K_ints + j_block];
                const uint32_t b1 = B[(p + 1) * K_ints + j_block];
                
                const float32x4_t va0_0 = vdupq_n_f32(rowA0[p]);
                const float32x4_t va0_1 = vdupq_n_f32(rowA0[p + 1]);
                
                float32x4_t va1_0, va1_1;
                if (rowA1) {
                    va1_0 = vdupq_n_f32(rowA1[p]);
                    va1_1 = vdupq_n_f32(rowA1[p + 1]);
                }

                // Interleave Row 0 and Row 1 calculations for better pipeline usage
                // p=0
                const uint8_t b0_0 = b0 & 0xFF;
                const uint8_t b0_1 = (b0 >> 8) & 0xFF;
                const uint8_t b0_2 = (b0 >> 16) & 0xFF;
                const uint8_t b0_3 = (b0 >> 24) & 0xFF;

                acc0[0] = vmlaq_f32(acc0[0], lut0[b0_0], va0_0);
                if (rowA1) acc1[0] = vmlaq_f32(acc1[0], lut0[b0_0], va1_0);
                acc0[1] = vmlaq_f32(acc0[1], lut1[b0_0], va0_0);
                if (rowA1) acc1[1] = vmlaq_f32(acc1[1], lut1[b0_0], va1_0);
                
                acc0[2] = vmlaq_f32(acc0[2], lut0[b0_1], va0_0);
                if (rowA1) acc1[2] = vmlaq_f32(acc1[2], lut0[b0_1], va1_0);
                acc0[3] = vmlaq_f32(acc0[3], lut1[b0_1], va0_0);
                if (rowA1) acc1[3] = vmlaq_f32(acc1[3], lut1[b0_1], va1_0);

                acc0[4] = vmlaq_f32(acc0[4], lut0[b0_2], va0_0);
                if (rowA1) acc1[4] = vmlaq_f32(acc1[4], lut0[b0_2], va1_0);
                acc0[5] = vmlaq_f32(acc0[5], lut1[b0_2], va0_0);
                if (rowA1) acc1[5] = vmlaq_f32(acc1[5], lut1[b0_2], va1_0);

                acc0[6] = vmlaq_f32(acc0[6], lut0[b0_3], va0_0);
                if (rowA1) acc1[6] = vmlaq_f32(acc1[6], lut0[b0_3], va1_0);
                acc0[7] = vmlaq_f32(acc0[7], lut1[b0_3], va0_0);
                if (rowA1) acc1[7] = vmlaq_f32(acc1[7], lut1[b0_3], va1_0);

                // p=1
                const uint8_t b1_0 = b1 & 0xFF;
                const uint8_t b1_1 = (b1 >> 8) & 0xFF;
                const uint8_t b1_2 = (b1 >> 16) & 0xFF;
                const uint8_t b1_3 = (b1 >> 24) & 0xFF;

                acc0[0] = vmlaq_f32(acc0[0], lut0[b1_0], va0_1);
                if (rowA1) acc1[0] = vmlaq_f32(acc1[0], lut0[b1_0], va1_1);
                acc0[1] = vmlaq_f32(acc0[1], lut1[b1_0], va0_1);
                if (rowA1) acc1[1] = vmlaq_f32(acc1[1], lut1[b1_0], va1_1);

                acc0[2] = vmlaq_f32(acc0[2], lut0[b1_1], va0_1);
                if (rowA1) acc1[2] = vmlaq_f32(acc1[2], lut0[b1_1], va1_1);
                acc0[3] = vmlaq_f32(acc0[3], lut1[b1_1], va0_1);
                if (rowA1) acc1[3] = vmlaq_f32(acc1[3], lut1[b1_1], va1_1);

                acc0[4] = vmlaq_f32(acc0[4], lut0[b1_2], va0_1);
                if (rowA1) acc1[4] = vmlaq_f32(acc1[4], lut0[b1_2], va1_1);
                acc0[5] = vmlaq_f32(acc0[5], lut1[b1_2], va0_1);
                if (rowA1) acc1[5] = vmlaq_f32(acc1[5], lut1[b1_2], va1_1);

                acc0[6] = vmlaq_f32(acc0[6], lut0[b1_3], va0_1);
                if (rowA1) acc1[6] = vmlaq_f32(acc1[6], lut0[b1_3], va1_1);
                acc0[7] = vmlaq_f32(acc0[7], lut1[b1_3], va0_1);
                if (rowA1) acc1[7] = vmlaq_f32(acc1[7], lut1[b1_3], va1_1);
            }

            float* out0 = &C[i * K + j_block * 32];
            for (int n = 0; n < 8; ++n) vst1q_f32(out0 + n * 4, acc0[n]);
            if (rowA1) {
                float* out1 = &C[(i + 1) * K + j_block * 32];
                for (int n = 0; n < 8; ++n) vst1q_f32(out1 + n * 4, acc1[n]);
            }
        }
    }
}
