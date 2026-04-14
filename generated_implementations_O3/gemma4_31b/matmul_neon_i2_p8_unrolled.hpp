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
                // Load A values to avoid repeatedly calling vdup inside the inner loop
                // This might help if the compiler isn't already hoisting them.
                const float a0_0 = rowA0[p], a0_1 = rowA0[p+1], a0_2 = rowA0[p+2], a0_3 = rowA0[p+3];
                const float a0_4 = rowA0[p+4], a0_5 = rowA0[p+5], a0_6 = rowA0[p+6], a0_7 = rowA0[p+7];
                
                float a1_0 = 0, a1_1 = 0, a1_2 = 0, a1_3 = 0, a1_4 = 0, a1_5 = 0, a1_6 = 0, a1_7 = 0;
                if (rowA1) {
                    a1_0 = rowA1[p]; a1_1 = rowA1[p+1]; a1_2 = rowA1[p+2]; a1_3 = rowA1[p+3];
                    a1_4 = rowA1[p+4]; a1_5 = rowA1[p+5]; a1_6 = rowA1[p+6]; a1_7 = rowA1[p+7];
                }

                const uint32_t p0 = B[p * K_ints + j_block];
                const uint32_t p1 = B[(p + 1) * K_ints + j_block];
                const uint32_t p2 = B[(p + 2) * K_ints + j_block];
                const uint32_t p3 = B[(p + 3) * K_ints + j_block];
                const uint32_t p4 = B[(p + 4) * K_ints + j_block];
                const uint32_t p5 = B[(p + 5) * K_ints + j_block];
                const uint32_t p6 = B[(p + 6) * K_ints + j_block];
                const uint32_t p7 = B[(p + 7) * K_ints + j_block];

                // Row 0 - Process p values
                const float32x4_t va0_0 = vdupq_n_f32(a0_0);
                acc0[0] = vmlaq_f32(acc0[0], sign_lut[p0 & 0xFF].v0, va0_0);
                acc0[1] = vmlaq_f32(acc0[1], sign_lut[p0 & 0xFF].v1, va0_0);
                acc0[2] = vmlaq_f32(acc0[2], sign_lut[(p0 >> 8) & 0xFF].v0, va0_0);
                acc0[3] = vmlaq_f32(acc0[3], sign_lut[(p0 >> 8) & 0xFF].v1, va0_0);
                acc0[4] = vmlaq_f32(acc0[4], sign_lut[(p0 >> 16) & 0xFF].v0, va0_0);
                acc0[5] = vmlaq_f32(acc0[5], sign_lut[(p0 >> 16) & 0xFF].v1, va0_0);
                acc0[6] = vmlaq_f32(acc0[6], sign_lut[(p0 >> 24) & 0xFF].v0, va0_0);
                acc0[7] = vmlaq_f32(acc0[7], sign_lut[(p0 >> 24) & 0xFF].v1, va0_0);

                const float32x4_t va0_1 = vdupq_n_f32(a0_1);
                acc0[0] = vmlaq_f32(acc0[0], sign_lut[p1 & 0xFF].v0, va0_1);
                acc0[1] = vmlaq_f32(acc0[1], sign_lut[p1 & 0xFF].v1, va0_1);
                acc0[2] = vmlaq_f32(acc0[2], sign_lut[(p1 >> 8) & 0xFF].v0, va0_1);
                acc0[3] = vmlaq_f32(acc0[3], sign_lut[(p1 >> 8) & 0xFF].v1, va0_1);
                acc0[4] = vmlaq_f32(acc0[4], sign_lut[(p1 >> 16) & 0xFF].v0, va0_1);
                acc0[5] = vmlaq_f32(acc0[5], sign_lut[(p1 >> 16) & 0xFF].v1, va0_1);
                acc0[6] = vmlaq_f32(acc0[6], sign_lut[(p1 >> 24) & 0xFF].v0, va0_1);
                acc0[7] = vmlaq_f32(acc0[7], sign_lut[(p1 >> 24) & 0xFF].v1, va0_1);

                const float32x4_t va0_2 = vdupq_n_f32(a0_2);
                acc0[0] = vmlaq_f32(acc0[0], sign_lut[p2 & 0xFF].v0, va0_2);
                acc0[1] = vmlaq_f32(acc0[1], sign_lut[p2 & 0xFF].v1, va0_2);
                acc0[2] = vmlaq_f32(acc0[2], sign_lut[(p2 >> 8) & 0xFF].v0, va0_2);
                acc0[3] = vmlaq_f32(acc0[3], sign_lut[(p2 >> 8) & 0xFF].v1, va0_2);
                acc0[4] = vmlaq_f32(acc0[4], sign_lut[(p2 >> 16) & 0xFF].v0, va0_2);
                acc0[5] = vmlaq_f32(acc0[5], sign_lut[(p2 >> 16) & 0xFF].v1, va0_2);
                acc0[6] = vmlaq_f32(acc0[6], sign_lut[(p2 >> 24) & 0xFF].v0, va0_2);
                acc0[7] = vmlaq_f32(acc0[7], sign_lut[(p2 >> 24) & 0xFF].v1, va0_2);

                const float32x4_t va0_3 = vdupq_n_f32(a0_3);
                acc0[0] = vmlaq_f32(acc0[0], sign_lut[p3 & 0xFF].v0, va0_3);
                acc0[1] = vmlaq_f32(acc0[1], sign_lut[p3 & 0xFF].v1, va0_3);
                acc0[2] = vmlaq_f32(acc0[2], sign_lut[(p3 >> 8) & 0xFF].v0, va0_3);
                acc0[3] = vmlaq_f32(acc0[3], sign_lut[(p3 >> 8) & 0xFF].v1, va0_3);
                acc0[4] = vmlaq_f32(acc0[4], sign_lut[(p3 >> 16) & 0xFF].v0, va0_3);
                acc0[5] = vmlaq_f32(acc0[5], sign_lut[(p3 >> 16) & 0xFF].v1, va0_3);
                acc0[6] = vmlaq_f32(acc0[6], sign_lut[(p3 >> 24) & 0xFF].v0, va0_3);
                acc0[7] = vmlaq_f32(acc0[7], sign_lut[(p3 >> 24) & 0xFF].v1, va0_3);

                const float32x4_t va0_4 = vdupq_n_f32(a0_4);
                acc0[0] = vmlaq_f32(acc0[0], sign_lut[p4 & 0xFF].v0, va0_4);
                acc0[1] = vmlaq_f32(acc0[1], sign_lut[p4 & 0xFF].v1, va0_4);
                acc0[2] = vmlaq_f32(acc0[2], sign_lut[(p4 >> 8) & 0xFF].v0, va0_4);
                acc0[3] = vmlaq_f32(acc0[3], sign_lut[(p4 >> 8) & 0xFF].v1, va0_4);
                acc0[4] = vmlaq_f32(acc0[4], sign_lut[(p4 >> 16) & 0xFF].v0, va0_4);
                acc0[5] = vmlaq_f32(acc0[5], sign_lut[(p4 >> 16) & 0xFF].v1, va0_4);
                acc0[6] = vmlaq_f32(acc0[6], sign_lut[(p4 >> 24) & 0xFF].v0, va0_4);
                acc0[7] = vmlaq_f32(acc0[7], sign_lut[(p4 >> 24) & 0xFF].v1, va0_4);

                const float32x4_t va0_5 = vdupq_n_f32(a0_5);
                acc0[0] = vmlaq_f32(acc0[0], sign_lut[p5 & 0xFF].v0, va0_5);
                acc0[1] = vmlaq_f32(acc0[1], sign_lut[p5 & 0xFF].v1, va0_5);
                acc0[2] = vmlaq_f32(acc0[2], sign_lut[(p5 >> 8) & 0xFF].v0, va0_5);
                acc0[3] = vmlaq_f32(acc0[3], sign_lut[(p5 >> 8) & 0xFF].v1, va0_5);
                acc0[4] = vmlaq_f32(acc0[4], sign_lut[(p5 >> 16) & 0xFF].v0, va0_5);
                acc0[5] = vmlaq_f32(acc0[5], sign_lut[(p5 >> 16) & 0xFF].v1, va0_5);
                acc0[6] = vmlaq_f32(acc0[6], sign_lut[(p5 >> 24) & 0xFF].v0, va0_5);
                acc0[7] = vmlaq_f32(acc0[7], sign_lut[(p5 >> 24) & 0xFF].v1, va0_5);

                const float32x4_t va0_6 = vdupq_n_f32(a0_6);
                acc0[0] = vmlaq_f32(acc0[0], sign_lut[p6 & 0xFF].v0, va0_6);
                acc0[1] = vmlaq_f32(acc0[1], sign_lut[p6 & 0xFF].v1, va0_6);
                acc0[2] = vmlaq_f32(acc0[2], sign_lut[(p6 >> 8) & 0xFF].v0, va0_6);
                acc0[3] = vmlaq_f32(acc0[3], sign_lut[(p6 >> 8) & 0xFF].v1, va0_6);
                acc0[4] = vmlaq_f32(acc0[4], sign_lut[(p6 >> 16) & 0xFF].v0, va0_6);
                acc0[5] = vmlaq_f32(acc0[5], sign_lut[(p6 >> 16) & 0xFF].v1, va0_6);
                acc0[6] = vmlaq_f32(acc0[6], sign_lut[(p6 >> 24) & 0xFF].v0, va0_6);
                acc0[7] = vmlaq_f32(acc0[7], sign_lut[(p6 >> 24) & 0xFF].v1, va0_6);

                const float32x4_t va0_7 = vdupq_n_f32(a0_7);
                acc0[0] = vmlaq_f32(acc0[0], sign_lut[p7 & 0xFF].v0, va0_7);
                acc0[1] = vmlaq_f32(acc0[1], sign_lut[p7 & 0xFF].v1, va0_7);
                acc0[2] = vmlaq_f32(acc0[2], sign_lut[(p7 >> 8) & 0xFF].v0, va0_7);
                acc0[3] = vmlaq_f32(acc0[3], sign_lut[(p7 >> 8) & 0xFF].v1, va0_7);
                acc0[4] = vmlaq_f32(acc0[4], sign_lut[(p7 >> 16) & 0xFF].v0, va0_7);
                acc0[5] = vmlaq_f32(acc0[5], sign_lut[(p7 >> 16) & 0xFF].v1, va0_7);
                acc0[6] = vmlaq_f32(acc0[6], sign_lut[(p7 >> 24) & 0xFF].v0, va0_7);
                acc0[7] = vmlaq_f32(acc0[7], sign_lut[(p7 >> 24) & 0xFF].v1, va0_7);

                if (rowA1) {
                    const float32x4_t va1_0 = vdupq_n_f32(a1_0);
                    acc1[0] = vmlaq_f32(acc1[0], sign_lut[p0 & 0xFF].v0, va1_0);
                    acc1[1] = vmlaq_f32(acc1[1], sign_lut[p0 & 0xFF].v1, va1_0);
                    acc1[2] = vmlaq_f32(acc1[2], sign_lut[(p0 >> 8) & 0xFF].v0, va1_0);
                    acc1[3] = vmlaq_f32(acc1[3], sign_lut[(p0 >> 8) & 0xFF].v1, va1_0);
                    acc1[4] = vmlaq_f32(acc1[4], sign_lut[(p0 >> 16) & 0xFF].v0, va1_0);
                    acc1[5] = vmlaq_f32(acc1[5], sign_lut[(p0 >> 16) & 0xFF].v1, va1_0);
                    acc1[6] = vmlaq_f32(acc1[6], sign_lut[(p0 >> 24) & 0xFF].v0, va1_0);
                    acc1[7] = vmlaq_f32(acc1[7], sign_lut[(p0 >> 24) & 0xFF].v1, va1_0);

                    const float32x4_t va1_1 = vdupq_n_f32(a1_1);
                    acc1[0] = vmlaq_f32(acc1[0], sign_lut[p1 & 0xFF].v0, va1_1);
                    acc1[1] = vmlaq_f32(acc1[1], sign_lut[p1 & 0xFF].v1, va1_1);
                    acc1[2] = vmlaq_f32(acc1[2], sign_lut[(p1 >> 8) & 0xFF].v0, va1_1);
                    acc1[3] = vmlaq_f32(acc1[3], sign_lut[(p1 >> 8) & 0xFF].v1, va1_1);
                    acc1[4] = vmlaq_f32(acc1[4], sign_lut[(p1 >> 16) & 0xFF].v0, va1_1);
                    acc1[5] = vmlaq_f32(acc1[5], sign_lut[(p1 >> 16) & 0xFF].v1, va1_1);
                    acc1[6] = vmlaq_f32(acc1[6], sign_lut[(p1 >> 24) & 0xFF].v0, va1_1);
                    acc1[7] = vmlaq_f32(acc1[7], sign_lut[(p1 >> 24) & 0xFF].v1, va1_1);

                    const float32x4_t va1_2 = vdupq_n_f32(a1_2);
                    acc1[0] = vmlaq_f32(acc1[0], sign_lut[p2 & 0xFF].v0, va1_2);
                    acc1[1] = vmlaq_f32(acc1[1], sign_lut[p2 & 0xFF].v1, va1_2);
                    acc1[2] = vmlaq_f32(acc1[2], sign_lut[(p2 >> 8) & 0xFF].v0, va1_2);
                    acc1[3] = vmlaq_f32(acc1[3], sign_lut[(p2 >> 8) & 0xFF].v1, va1_2);
                    acc1[4] = vmlaq_f32(acc1[4], sign_lut[(p2 >> 16) & 0xFF].v0, va1_2);
                    acc1[5] = vmlaq_f32(acc1[5], sign_lut[(p2 >> 16) & 0xFF].v1, va1_2);
                    acc1[6] = vmlaq_f32(acc1[6], sign_lut[(p2 >> 24) & 0xFF].v0, va1_2);
                    acc1[7] = vmlaq_f32(acc1[7], sign_lut[(p2 >> 24) & 0xFF].v1, va1_2);

                    const float32x4_t va1_3 = vdupq_n_f32(a1_3);
                    acc1[0] = vmlaq_f32(acc1[0], sign_lut[p3 & 0xFF].v0, va1_3);
                    acc1[1] = vmlaq_f32(acc1[1], sign_lut[p3 & 0xFF].v1, va1_3);
                    acc1[2] = vmlaq_f32(acc1[2], sign_lut[(p3 >> 8) & 0xFF].v0, va1_3);
                    acc1[3] = vmlaq_f32(acc1[3], sign_lut[(p3 >> 8) & 0xFF].v1, va1_3);
                    acc1[4] = vmlaq_f32(acc1[4], sign_lut[(p3 >> 16) & 0xFF].v0, va1_3);
                    acc1[5] = vmlaq_f32(acc1[5], sign_lut[(p3 >> 16) & 0xFF].v1, va1_3);
                    acc1[6] = vmlaq_f32(acc1[6], sign_lut[(p3 >> 24) & 0xFF].v0, va1_3);
                    acc1[7] = vmlaq_f32(acc1[7], sign_lut[(p3 >> 24) & 0xFF].v1, va1_3);

                    const float32x4_t va1_4 = vdupq_n_f32(a1_4);
                    acc1[0] = vmlaq_f32(acc1[0], sign_lut[p4 & 0xFF].v0, va1_4);
                    acc1[1] = vmlaq_f32(acc1[1], sign_lut[p4 & 0xFF].v1, va1_4);
                    acc1[2] = vmlaq_f32(acc1[2], sign_lut[(p4 >> 8) & 0xFF].v0, va1_4);
                    acc1[3] = vmlaq_f32(acc1[3], sign_lut[(p4 >> 8) & 0xFF].v1, va1_4);
                    acc1[4] = vmlaq_f32(acc1[4], sign_lut[(p4 >> 16) & 0xFF].v0, va1_4);
                    acc1[5] = vmlaq_f32(acc1[5], sign_lut[(p4 >> 16) & 0xFF].v1, va1_4);
                    acc1[6] = vmlaq_f32(acc1[6], sign_lut[(p4 >> 24) & 0xFF].v0, va1_4);
                    acc1[7] = vmlaq_f32(acc1[7], sign_lut[(p4 >> 24) & 0xFF].v1, va1_4);

                    const float32x4_t va1_5 = vdupq_n_f32(a1_5);
                    acc1[0] = vmlaq_f32(acc1[0], sign_lut[p5 & 0xFF].v0, va1_5);
                    acc1[1] = vmlaq_f32(acc1[1], sign_lut[p5 & 0xFF].v1, va1_5);
                    acc1[2] = vmlaq_f32(acc1[2], sign_lut[(p5 >> 8) & 0xFF].v0, va1_5);
                    acc1[3] = vmlaq_f32(acc1[3], sign_lut[(p5 >> 8) & 0xFF].v1, va1_5);
                    acc1[4] = vmlaq_f32(acc1[4], sign_lut[(p5 >> 16) & 0xFF].v0, va1_5);
                    acc1[5] = vmlaq_f32(acc1[5], sign_lut[(p5 >> 16) & 0xFF].v1, va1_5);
                    acc1[6] = vmlaq_f32(acc1[6], sign_lut[(p5 >> 24) & 0xFF].v0, va1_5);
                    acc1[7] = vmlaq_f32(acc1[7], sign_lut[(p5 >> 24) & 0xFF].v1, va1_5);

                    const float32x4_t va1_6 = vdupq_n_f32(a1_6);
                    acc1[0] = vmlaq_f32(acc1[0], sign_lut[p6 & 0xFF].v0, va1_6);
                    acc1[1] = vmlaq_f32(acc1[1], sign_lut[p6 & 0xFF].v1, va1_6);
                    acc1[2] = vmlaq_f32(acc1[2], sign_lut[(p6 >> 8) & 0xFF].v0, va1_6);
                    acc1[3] = vmlaq_f32(acc1[3], sign_lut[(p6 >> 8) & 0xFF].v1, va1_6);
                    acc1[4] = vmlaq_f32(acc1[4], sign_lut[(p6 >> 16) & 0xFF].v0, va1_6);
                    acc1[5] = vmlaq_f32(acc1[5], sign_lut[(p6 >> 16) & 0xFF].v1, va1_6);
                    acc1[6] = vmlaq_f32(acc1[6], sign_lut[(p6 >> 24) & 0xFF].v0, va1_6);
                    acc1[7] = vmlaq_f32(acc1[7], sign_lut[(p6 >> 24) & 0xFF].v1, va1_6);

                    const float32x4_t va1_7 = vdupq_n_f32(a1_7);
                    acc1[0] = vmlaq_f32(acc1[0], sign_lut[p7 & 0xFF].v0, va1_7);
                    acc1[1] = vmlaq_f32(acc1[1], sign_lut[p7 & 0xFF].v1, va1_7);
                    acc1[2] = vmlaq_f32(acc1[2], sign_lut[(p7 >> 8) & 0xFF].v0, va1_7);
                    acc1[3] = vmlaq_f32(acc1[3], sign_lut[(p7 >> 8) & 0xFF].v1, va1_7);
                    acc1[4] = vmlaq_f32(acc1[4], sign_lut[(p7 >> 16) & 0xFF].v0, va1_7);
                    acc1[5] = vmlaq_f32(acc1[5], sign_lut[(p7 >> 16) & 0xFF].v1, va1_7);
                    acc1[6] = vmlaq_f32(acc1[6], sign_lut[(p7 >> 24) & 0xFF].v0, va1_7);
                    acc1[7] = vmlaq_f32(acc1[7], sign_lut[(p7 >> 24) & 0xFF].v1, va1_7);
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
