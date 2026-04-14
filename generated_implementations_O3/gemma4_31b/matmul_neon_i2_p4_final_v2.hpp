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

    size_t i = 0;
    // Main loop processing 2 rows at a time
    for (; i + 1 < M; i += 2) {
        const float* rowA0 = &A[i * K];
        const float* rowA1 = &A[(i + 1) * K];
        
        for (size_t j_block = 0; j_block < K_ints; ++j_block) {
            float32x4_t acc0[8], acc1[8];
            for (int n = 0; n < 8; ++n) {
                acc0[n] = vdupq_n_f32(0.0f);
                acc1[n] = vdupq_n_f32(0.0f);
            }

            for (size_t p = 0; p < K; p += 4) {
                // Pre-load A values
                const float32x4_t va0_0 = vdupq_n_f32(rowA0[p]);
                const float32x4_t va0_1 = vdupq_n_f32(rowA0[p+1]);
                const float32x4_t va0_2 = vdupq_n_f32(rowA0[p+2]);
                const float32x4_t va0_3 = vdupq_n_f32(rowA0[p+3]);
                
                const float32x4_t va1_0 = vdupq_n_f32(rowA1[p]);
                const float32x4_t va1_1 = vdupq_n_f32(rowA1[p+1]);
                const float32x4_t va1_2 = vdupq_n_f32(rowA1[p+2]);
                const float32x4_t va1_3 = vdupq_n_f32(rowA1[p+3]);

                // Process 4 elements of the dot product
                // Element 0
                const uint32_t b0 = B[p * K_ints + j_block];
                acc0[0] = vmlaq_f32(acc0[0], sign_lut[b0 & 0xFF].v0, va0_0);
                acc0[1] = vmlaq_f32(acc0[1], sign_lut[b0 & 0xFF].v1, va0_0);
                acc0[2] = vmlaq_f32(acc0[2], sign_lut[(b0 >> 8) & 0xFF].v0, va0_0);
                acc0[3] = vmlaq_f32(acc0[3], sign_lut[(b0 >> 8) & 0xFF].v1, va0_0);
                acc0[4] = vmlaq_f32(acc0[4], sign_lut[(b0 >> 16) & 0xFF].v0, va0_0);
                acc0[5] = vmlaq_f32(acc0[5], sign_lut[(b0 >> 16) & 0xFF].v1, va0_0);
                acc0[6] = vmlaq_f32(acc0[6], sign_lut[(b0 >> 24) & 0xFF].v0, va0_0);
                acc0[7] = vmlaq_f32(acc0[7], sign_lut[(b0 >> 24) & 0xFF].v1, va0_0);
                acc1[0] = vmlaq_f32(acc1[0], sign_lut[b0 & 0xFF].v0, va1_0);
                acc1[1] = vmlaq_f32(acc1[1], sign_lut[b0 & 0xFF].v1, va1_0);
                acc1[2] = vmlaq_f32(acc1[2], sign_lut[(b0 >> 8) & 0xFF].v0, va1_0);
                acc1[3] = vmlaq_f32(acc1[3], sign_lut[(b0 >> 8) & 0xFF].v1, va1_0);
                acc1[4] = vmlaq_f32(acc1[4], sign_lut[(b0 >> 16) & 0xFF].v0, va1_0);
                acc1[5] = vmlaq_f32(acc1[5], sign_lut[(b0 >> 16) & 0xFF].v1, va1_0);
                acc1[6] = vmlaq_f32(acc1[6], sign_lut[(b0 >> 24) & 0xFF].v0, va1_0);
                acc1[7] = vmlaq_f32(acc1[7], sign_lut[(b0 >> 24) & 0xFF].v1, va1_0);

                // Element 1
                const uint32_t b1 = B[(p + 1) * K_ints + j_block];
                acc0[0] = vmlaq_f32(acc0[0], sign_lut[b1 & 0xFF].v0, va0_1);
                acc0[1] = vmlaq_f32(acc0[1], sign_lut[b1 & 0xFF].v1, va0_1);
                acc0[2] = vmlaq_f32(acc0[2], sign_lut[(b1 >> 8) & 0xFF].v0, va0_1);
                acc0[3] = vmlaq_f32(acc0[3], sign_lut[(b1 >> 8) & 0xFF].v1, va0_1);
                acc0[4] = vmlaq_f32(acc0[4], sign_lut[(b1 >> 16) & 0xFF].v0, va0_1);
                acc0[5] = vmlaq_f32(acc0[5], sign_lut[(b1 >> 16) & 0xFF].v1, va0_1);
                acc0[6] = vmlaq_f32(acc0[6], sign_lut[(b1 >> 24) & 0xFF].v0, va0_1);
                acc0[7] = vmlaq_f32(acc0[7], sign_lut[(b1 >> 24) & 0xFF].v1, va0_1);
                acc1[0] = vmlaq_f32(acc1[0], sign_lut[b1 & 0xFF].v0, va1_1);
                acc1[1] = vmlaq_f32(acc1[1], sign_lut[b1 & 0xFF].v1, va1_1);
                acc1[2] = vmlaq_f32(acc1[2], sign_lut[(b1 >> 8) & 0xFF].v0, va1_1);
                acc1[3] = vmlaq_f32(acc1[3], sign_lut[(b1 >> 8) & 0xFF].v1, va1_1);
                acc1[4] = vmlaq_f32(acc1[4], sign_lut[(b1 >> 16) & 0xFF].v0, va1_1);
                acc1[5] = vmlaq_f32(acc1[5], sign_lut[(b1 >> 16) & 0xFF].v1, va1_1);
                acc1[6] = vmlaq_f32(acc1[6], sign_lut[(b1 >> 24) & 0xFF].v0, va1_1);
                acc1[7] = vmlaq_f32(acc1[7], sign_lut[(b1 >> 24) & 0xFF].v1, va1_1);

                // Element 2
                const uint32_t b2 = B[(p + 2) * K_ints + j_block];
                acc0[0] = vmlaq_f32(acc0[0], sign_lut[b2 & 0xFF].v0, va0_2);
                acc0[1] = vmlaq_f32(acc0[1], sign_lut[b2 & 0xFF].v1, va0_2);
                acc0[2] = vmlaq_f32(acc0[2], sign_lut[(b2 >> 8) & 0xFF].v0, va0_2);
                acc0[3] = vmlaq_f32(acc0[3], sign_lut[(b2 >> 8) & 0xFF].v1, va0_2);
                acc0[4] = vmlaq_f32(acc0[4], sign_lut[(b2 >> 16) & 0xFF].v0, va0_2);
                acc0[5] = vmlaq_f32(acc0[5], sign_lut[(b2 >> 16) & 0xFF].v1, va0_2);
                acc0[6] = vmlaq_f32(acc0[6], sign_lut[(b2 >> 24) & 0xFF].v0, va0_2);
                acc0[7] = vmlaq_f32(acc0[7], sign_lut[(b2 >> 24) & 0xFF].v1, va0_2);
                acc1[0] = vmlaq_f32(acc1[0], sign_lut[b2 & 0xFF].v0, va1_2);
                acc1[1] = vmlaq_f32(acc1[1], sign_lut[b2 & 0xFF].v1, va1_2);
                acc1[2] = vmlaq_f32(acc1[2], sign_lut[(b2 >> 8) & 0xFF].v0, va1_2);
                acc1[3] = vmlaq_f32(acc1[3], sign_lut[(b2 >> 8) & 0xFF].v1, va1_2);
                acc1[4] = vmlaq_f32(acc1[4], sign_lut[(b2 >> 16) & 0xFF].v0, va1_2);
                acc1[5] = vmlaq_f32(acc1[5], sign_lut[(b2 >> 16) & 0xFF].v1, va1_2);
                acc1[6] = vmlaq_f32(acc1[6], sign_lut[(b2 >> 24) & 0xFF].v0, va1_2);
                acc1[7] = vmlaq_f32(acc1[7], sign_lut[(b2 >> 24) & 0xFF].v1, va1_2);

                // Element 3
                const uint32_t b3 = B[(p + 3) * K_ints + j_block];
                acc0[0] = vmlaq_f32(acc0[0], sign_lut[b3 & 0xFF].v0, va0_3);
                acc0[1] = vmlaq_f32(acc0[1], sign_lut[b3 & 0xFF].v1, va0_3);
                acc0[2] = vmlaq_f32(acc0[2], sign_lut[(b3 >> 8) & 0xFF].v0, va0_3);
                acc0[3] = vmlaq_f32(acc0[3], sign_lut[(b3 >> 8) & 0xFF].v1, va0_3);
                acc0[4] = vmlaq_f32(acc0[4], sign_lut[(b3 >> 16) & 0xFF].v0, va0_3);
                acc0[5] = vmlaq_f32(acc0[5], sign_lut[(b3 >> 16) & 0xFF].v1, va0_3);
                acc0[6] = vmlaq_f32(acc0[6], sign_lut[(b3 >> 24) & 0xFF].v0, va0_3);
                acc0[7] = vmlaq_f32(acc0[7], sign_lut[(b3 >> 24) & 0xFF].v1, va0_3);
                acc1[0] = vmlaq_f32(acc1[0], sign_lut[b3 & 0xFF].v0, va1_3);
                acc1[1] = vmlaq_f32(acc1[1], sign_lut[b3 & 0xFF].v1, va1_3);
                acc1[2] = vmlaq_f32(acc1[2], sign_lut[(b3 >> 8) & 0xFF].v0, va1_3);
                acc1[3] = vmlaq_f32(acc1[3], sign_lut[(b3 >> 8) & 0xFF].v1, va1_3);
                acc1[4] = vmlaq_f32(acc1[4], sign_lut[(b3 >> 16) & 0xFF].v0, va1_3);
                acc1[5] = vmlaq_f32(acc1[5], sign_lut[(b3 >> 16) & 0xFF].v1, va1_3);
                acc1[6] = vmlaq_f32(acc1[6], sign_lut[(b3 >> 24) & 0xFF].v0, va1_3);
                acc1[7] = vmlaq_f32(acc1[7], sign_lut[(b3 >> 24) & 0xFF].v1, va1_3);
            }

            float* out0 = &C[i * K + j_block * 32];
            for (int n = 0; n < 8; ++n) vst1q_f32(out0 + n * 4, acc0[n]);
            float* out1 = &C[(i + 1) * K + j_block * 32];
            for (int n = 0; n < 8; ++n) vst1q_f32(out1 + n * 4, acc1[n]);
        }
    }

    // Handle remaining row
    if (i < M) {
        const float* rowA = &A[i * K];
        for (size_t j_block = 0; j_block < K_ints; ++j_block) {
            float32x4_t acc[8];
            for (int n = 0; n < 8; ++n) acc[n] = vdupq_n_f32(0.0f);
            for (size_t p = 0; p < K; ++p) {
                const float32x4_t va = vdupq_n_f32(rowA[p]);
                const uint32_t b = B[p * K_ints + j_block];
                acc[0] = vmlaq_f32(acc[0], sign_lut[b & 0xFF].v0, va);
                acc[1] = vmlaq_f32(acc[1], sign_lut[b & 0xFF].v1, va);
                acc[2] = vmlaq_f32(acc[2], sign_lut[(b >> 8) & 0xFF].v0, va);
                acc[3] = vmlaq_f32(acc[3], sign_lut[(b >> 8) & 0xFF].v1, va);
                acc[4] = vmlaq_f32(acc[4], sign_lut[(b >> 16) & 0xFF].v0, va);
                acc[5] = vmlaq_f32(acc the 5, sign_lut[(b >> 16) & 0xFF].v1, va);
                acc[6] = vmlaq_f32(acc[6], sign_lut[(b >> 24) & 0xFF].v0, va);
                acc[7] = vmlaq_f32(acc[7], sign_lut[(b >> 24) & 0xFF].v1, va);
            }
            float* out = &C[i * K + j_block * 32];
            for (int n = 0; n < 8; ++n) vst1q_f32(out + n * 4, acc[n]);
        }
    }
}
