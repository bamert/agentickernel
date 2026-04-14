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
    // Process rows in pairs.
    while (i + 1 < M) {
        const float* rowA0 = &A[i * K];
        const float* rowA1 = &A[(i + 1) * K];
        
        for (size_t j_block = 0; j_block < K_ints; ++j_block) {
            float32x4_t acc0[8], acc1[8];
            for (int n = 0; n < 8; ++n) {
                acc0[n] = vdupq_n_f32(0.0f);
                acc1[n] = vdupq_n_f32(0.0f);
            }

            const uint32_t* b_ptr = &B[j_block]; // Start at the first row's j_block
            
            for (size_t p = 0; p < K; ++p) {
                const uint32_t p0 = *b_ptr;
                const float32x4_t va0 = vdupq_n_f32(rowA0[p]);
                const float32x4_t va1 = vdupq_n_f32(rowA1[p]);

                // Row 0
                acc0[0] = vmlaq_f32(acc0[0], sign_lut[p0 & 0xFF].v0, va0);
                acc0[1] = vmlaq_f32(acc0[1], sign_lut[p0 & 0xFF].v1, va0);
                acc0[2] = vmlaq_f32(acc0[2], sign_lut[(p0 >> 8) & 0xFF].v0, va0);
                acc0[3] = vmlaq_f32(acc0[3], sign_lut[(p0 >> 8) & 0xFF].v1, va0);
                acc0[4] = vmlaq_f32(acc0[4], sign_lut[(p0 >> 16) & 0xFF].v0, va0);
                acc0[5] = vmlaq_f32(acc0[5], sign_lut[(p0 >> 16) & 0xFF].v1, va0);
                acc0[6] = vmlaq_f32(acc0[6], sign_lut[(p0 >> 24) & 0xFF].v0, va0);
                acc0[7] = vmlaq_f32(acc0[7], sign_lut[(p0 >> 24) & 0xFF].v1, va0);

                // Row 1
                acc1[0] = vmlaq_f32(acc1[0], sign_lut[p0 & 0xFF].v0, va1);
                acc1[1] = vmlaq_f32(acc1[1], sign_lut[p0 & 0xFF].v1, va1);
                acc1[2] = vmlaq_f32(acc1[2], sign_lut[(p0 >> 8) & 0xFF].v0, va1);
                acc1[3] = vmlaq_f32(acc1[3], sign_lut[(p0 >> 8) & 0xFF].v1, va1);
                acc1[4] = vmlaq_f32(acc1[4], sign_lut[(p0 >> 16) & 0xFF].v0, va1);
                acc1[5] = vmlaq_f32(acc1[5], sign_lut[(p0 >> 16) & 0xFF].v1, va1);
                acc1[6] = vmlaq_f32(acc1[6], sign_lut[(p0 >> 24) & 0xFF].v0, va1);
                acc1[7] = vmlaq_f32(acc1[7], sign_lut[(p0 >> 24) & 0xFF].v1, va1);

                b_ptr += K_ints;
            }

            float* out0 = &C[i * K + j_block * 32];
            float* out1 = &C[(i + 1) * K + j_block * 32];
            for (int n = 0; n < 8; ++n) {
                vst1q_f32(out0 + n * 4, acc0[n]);
                vst1q_f32(out1 + n * 4, acc1[n]);
            }
        }
        i += 2;
    }

    // Final row
    if (i < M) {
        const float* rowA = &A[i * K];
        for (size_t j_block = 0; j_block < K_ints; ++j_block) {
            float32x4_t acc[8];
            for (int n = 0; n < 8; ++n) acc[n] = vdupq_n_f32(0.0f);
            const uint32_t* b_ptr = &B[j_block];
            for (size_t p = 0; p < K; ++p) {
                const uint32_t p0 = *b_ptr;
                const float32x4_t va = vdupq_n_f32(rowA[p]);
                acc[0] = vmlaq_f32(acc[0], sign_lut[p0 & 0xFF].v0, va);
                acc[1] = vmlaq_f32(acc[1], sign_lut[p0 & 0xFF].v1, va);
                acc[2] = vmlaq_f32(acc[2], sign_lut[(p0 >> 8) & 0xFF].v0, va);
                acc[3] = vmlaq_f32(acc[3], sign_lut[(p0 >> 8) & 0xFF].v1, va);
                acc[4] = vmlaq_f32(acc[4], sign_lut[(p0 >> 16) & 0xFF].v0, va);
                acc[5] = vmlaq_f32(acc[5], sign_lut[(p0 >> 16) & 0xFF].v1, va);
                acc[6] = vmlaq_f32(acc[6], sign_lut[(p0 >> 24) & 0xFF].v0, va);
                acc[7] = vmlaq_f32(acc[7], sign_lut[(p0 >> 24) & 0xFF].v1, va);
                b_ptr += K_ints;
            }
            float* out = &C[i * K + j_block * 32];
            for (int n = 0; n < 8; ++n) vst1q_f32(out + n * 4, acc[n]);
        }
    }
}
