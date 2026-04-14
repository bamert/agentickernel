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

    for (size_t i = 0; i < M; i += 2) {
        const float* rowA0 = &A[i * K];
        const float* rowA1 = (i + 1 < M) ? &A[(i + 1) * K] : nullptr;
        
        for (size_t j_block = 0; j_block < K_ints; ++j_block) {
            float32x4_t acc0[8], acc1[8];
            for(int n=0; n<8; ++n) {
                acc0[n] = vdupq_n_f32(0.0f);
                acc1[n] = vdupq_n_f32(0.0f);
            }

            for (size_t p = 0; p < K; p += 8) {
                // Load B values for these 8 p's
                uint32_t bp[8];
                for(int n=0; n<8; ++n) bp[n] = B[(p + n) * K_ints + j_block];

                // Process each p in the unrolled block
                for(int n=0; n<8; ++n) {
                    const float32x4_t va0 = vdupq_n_f32(rowA0[p + n]);
                    const uint32_t val = bp[n];
                    
                    acc0[0] = vmlaq_f32(acc0[0], sign_lut[val & 0xFF].v0, va0);
                    acc0[1] = vmlaq_f32(acc0[1], sign_lut[val & 0xFF].v1, va0);
                    acc0[2] = vmlaq_f32(acc0[2], sign_lut[(val >> 8) & 0xFF].v0, va0);
                    acc0[3] = vmlaq_f32(acc0[3], sign_lut[(val >> 8) & 0xFF].v1, va0);
                    acc0[4] = vmlaq_f32(acc0[4], sign_lut[(val >> 16) & 0xFF].v0, va0);
                    acc0[5] = vmlaq_f32(acc0[5], sign_lut[(val >> 16) & 0xFF].v1, va0);
                    acc0[6] = vmlaq_f32(acc0[6], sign_lut[(val >> 24) & 0xFF].v0, va0);
                    acc0[7] = vmlaq_f32(acc0[7], sign_lut[(val >> 24) & 0xFF].v1, va0);

                    if (rowA1) {
                        const float32x4_t va1 = vdupq_n_f32(rowA1[p + n]);
                        acc1[0] = vmlaq_f32(acc1[0], sign_lut[val & 0xFF].v0, va1);
                        acc1[1] = vmlaq_f32(acc1[1], sign_lut[val & 0xFF].v1, va1);
                        acc1[2] = vmlaq_f32(acc1[2], sign_lut[(val >> 8) & 0xFF].v0, va1);
                        acc1[3] = vmlaq_f32(acc1[3], sign_lut[(val >> 8) & 0xFF].v1, va1);
                        acc1[4] = vmlaq_f32(acc1[4], sign_lut[(val >> 16) & 0xFF].v0, va1);
                        acc1[5] = vmlaq_f32(acc1[5], sign_lut[(val >> 16) & 0xFF].v1, va1);
                        acc1[6] = vmlaq_f32(acc1[6], sign_lut[(val >> 24) & 0xFF].v0, va1);
                        acc1[7] = vmlaq_f32(acc1[7], sign_lut[(val >> 24) & 0xFF].v1, va1);
                    }
                }
            }

            float* out_ptr0 = &C[i * K + j_block * 32];
            for(int n=0; n<8; ++n) vst1q_f32(out_ptr0 + n*4, acc0[n]);
            if (i + 1 < M) {
                float* out_ptr1 = &C[(i + 1) * K + j_block * 32];
                for(int n=0; n<8; ++n) vst1q_f32(out_ptr1 + n*4, acc1[n]);
            }
        }
    }
}
