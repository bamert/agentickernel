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

        for (size_t j_block = 0; j_block < K_ints; j_block += 4) {
            // Accumulators for 4 blocks of 32 elements (4 * 8 = 32 vectors)
            float32x4_t acc0[8], acc1[8], acc2[8], acc3[8];
            for(int n=0; n<8; ++n) {
                acc0[n] = vdupq_n_f32(0.0f);
                acc1[n] = vdupq_n_f32(0.0f);
                acc2[n] = vdupq_n_f32(0.0f);
                acc3[n] = vdupq_n_f32(0.0f);
            }

            for (size_t p = 0; p < K; p += 8) {
                const float32x4_t v_a[8];
                for(int n=0; n<8; ++n) v_a[n] = vdupq_n_f32(rowA[p + n]);

                const uint32_t p0[8], p1[8], p2[8], p3[8];
                for(int n=0; n<8; ++n) {
                    p0[n] = B[(p + n) * K_ints + j_block];
                    if (j_block + 1 < K_ints) p1[n] = B[(p + n) * K_ints + j_block + 1];
                    if (j_block + 2 < K_ints) p2[n] = B[(p + n) * K_ints + j_block + 2];
                    if (j_block + 3 < K_ints) p3[n] = B[(p + n) * K_ints + j_block + 3];
                }

                for(int n=0; n<8; ++n) {
                    const uint32_t val0 = p0[n];
                    acc0[0] = vmlaq_f32(acc0[0], sign_lut[val0 & 0xFF].v0, v_a[n]);
                    acc0[1] = vmlaq_f32(acc0[1], sign_lut[val0 & 0xFF].v1, v_a[n]);
                    acc0[2] = vmlaq_f32(acc0[2], sign_lut[(val0 >> 8) & 0xFF].v0, v_a[n]);
                    acc0[3] = vmlaq_f32(acc0[3], sign_lut[(val0 >> 8) & 0xFF].v1, v_a[n]);
                    acc0[4] = vmlaq_f32(acc0[4], sign_lut[(val0 >> 16) & 0xFF].v0, v_a[n]);
                    acc0[5] = vmlaq_f32(acc0[5], sign_lut[(val0 >> 16) & 0xFF].v1, v_a[n]);
                    acc0[6] = vmlaq_f32(acc0[6], sign_lut[(val0 >> 24) & 0xFF].v0, v_a[n]);
                    acc0[7] = vmlaq_f32(acc0[7], sign_lut[(val0 >> 24) & 0xFF].v1, v_a[n]);

                    if (j_block + 1 < K_ints) {
                        const uint32_t val1 = p1[n];
                        acc1[0] = vmlaq_f32(acc1[0], sign_lut[val1 & 0xFF].v0, v_a[n]);
                        acc1[1] = vmlaq_f32(acc1[1], sign_lut[val1 & 0xFF].v1, v_a[n]);
                        acc1[2] = vmlaq_f32(acc1[2], sign_lut[(val1 >> 8) & 0xFF].v0, v_a[n]);
                        acc1[3] = vmlaq_f32(acc1[3], sign_lut[(val1 >> 8) & 0xFF].v1, v_a[n]);
                        acc1[4] = vmlaq_f32(acc1[4], sign_lut[(val1 >> 16) & 0xFF].v0, v_a[n]);
                        acc1[5] = vmlaq_f32(acc1[5], sign_lut[(val1 >> 16) & 0xFF].v1, v_a[n]);
                        acc1[6] = vmlaq_f32(acc1[6], sign_lut[(val1 >> 24) & 0xFF].v0, v_a[n]);
                        acc1[7] = vmlaq_f32(acc1[7], sign_lut[(val1 >> 24) & 0xFF].v1, v_a[n]);
                    }
                    if (j_block + 2 < K_ints) {
                        const uint32_t val2 = p2[n];
                        acc2[0] = vmlaq_f32(acc2[0], sign_lut[val2 & 0xFF].v0, v_a[n]);
                        acc2[1] = vmlaq_f32(acc2[1], sign_lut[val2 & 0xFF].v1, v_a[n]);
                        acc2[2] = vmlaq_f32(acc2[2], sign_lut[(val2 >> 8) & 0xFF].v0, v_a[n]);
                        acc2[3] = vmlaq_f32(acc2[3], sign_lut[(val2 >> 8) & 0xFF].v1, v_a[n]);
                        acc2[4] = vmlaq_f32(acc2[4], sign_lut[(val2 >> 16) & 0xFF].v0, v_a[n]);
                        acc2[5] = vmlaq_f32(acc2[5], sign_lut[(val2 >> 16) & 0xFF].v1, v_a[n]);
                        acc2[6] = vmlaq_f32(acc2[6], sign_lut[(val2 >> 24) & 0xFF].v0, v_a[n]);
                        acc2[7] = vmlaq_f32(acc2[7], sign_lut[(val2 >> 24) & 0xFF].v1, v_a[n]);
                    }
                    if (j_block + 3 < K_ints) {
                        const uint32_t val3 = p3[n];
                        acc3[0] = vmlaq_f32(acc3[0], sign_lut[val3 & 0xFF].v0, v_a[n]);
                        acc3[1] = vmlaq_f32(acc3[1], sign_lut[val3 & 0xFF].v1, v_a[n]);
                        acc3[2] = vmlaq_f32(acc3[2], sign_lut[(val3 >> 8) & 0xFF].v0, v_a[n]);
                        acc3[3] = vmlaq_f32(acc3[3], sign_lut[(val3 >> 8) & 0xFF].v1, v_a[n]);
                        acc3[4] = vmlaq_f32(acc3[4], sign_lut[(val3 >> 16) & 0xFF].v0, v_a[n]);
                        acc3[5] = vmlaq_f32(acc3[5], sign_lut[(val3 >> 16) & 0xFF].v1, v_a[n]);
                        acc3[6] = vmlaq_f32(acc3[6], sign_lut[(val3 >> 24) & 0xFF].v0, v_a[n]);
                        acc3[7] = vmlaq_f32(acc3[7], sign_lut[(val3 >> 24) & 0xFF].v1, v_a[n]);
                    }
                }
            }
            
            for(int b=0; b<8; ++b) vst1q_f32(&rowC[j_block * 32 + b*4], acc0[b]);
            if (j_block + 1 < K_ints) for(int b=0; b<8; ++b) vst1q_f32(&rowC[(j_block + 1) * 32 + b*4], acc1[b]);
            if (j_block + 2 < K_ints) for(int b=0; b<8; ++b) vst1q_f32(&rowC[(j_block + 2) * 32 + b*4], acc2[b]);
            if (j_block + 3 < K_ints) for(int b=0; b<8; ++b) vst1q_f32(&rowC[(j_block + 3) * 32 + b*4], acc3[b]);
        }
    }
}
