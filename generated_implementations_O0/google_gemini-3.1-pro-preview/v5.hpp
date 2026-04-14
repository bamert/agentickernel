#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <arm_neon.h>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    std::vector<float> row_sums(M, 0.0f);
    for (size_t i = 0; i < M; ++i) {
        float sum = 0;
        for (size_t p = 0; p < K; ++p) {
            sum += A[i * K + p];
        }
        row_sums[i] = sum;
    }

    int32x4_t sh0 = {31, 30, 29, 28};
    int32x4_t sh1 = {27, 26, 25, 24};
    int32x4_t sh2 = {23, 22, 21, 20};
    int32x4_t sh3 = {19, 18, 17, 16};
    int32x4_t sh4 = {15, 14, 13, 12};
    int32x4_t sh5 = {11, 10,  9,  8};
    int32x4_t sh6 = { 7,  6,  5,  4};
    int32x4_t sh7 = { 3,  2,  1,  0};

    size_t i = 0;
    for (; i + 1 < M; i += 2) {
        for (size_t j_int = 0; j_int < K_ints; ++j_int) {
            float32x4_t c0_0 = vdupq_n_f32(0); float32x4_t c0_1 = vdupq_n_f32(0); float32x4_t c0_2 = vdupq_n_f32(0); float32x4_t c0_3 = vdupq_n_f32(0);
            float32x4_t c0_4 = vdupq_n_f32(0); float32x4_t c0_5 = vdupq_n_f32(0); float32x4_t c0_6 = vdupq_n_f32(0); float32x4_t c0_7 = vdupq_n_f32(0);

            float32x4_t c1_0 = vdupq_n_f32(0); float32x4_t c1_1 = vdupq_n_f32(0); float32x4_t c1_2 = vdupq_n_f32(0); float32x4_t c1_3 = vdupq_n_f32(0);
            float32x4_t c1_4 = vdupq_n_f32(0); float32x4_t c1_5 = vdupq_n_f32(0); float32x4_t c1_6 = vdupq_n_f32(0); float32x4_t c1_7 = vdupq_n_f32(0);

            for (size_t p = 0; p < K; ++p) {
                uint32_t packed = B[p * K_ints + j_int];
                uint32x4_t p_vec = vdupq_n_u32(packed);

                uint32x4_t a0 = vreinterpretq_u32_f32(vdupq_n_f32(A[(i + 0) * K + p]));
                uint32x4_t a1 = vreinterpretq_u32_f32(vdupq_n_f32(A[(i + 1) * K + p]));

                // Bit 0..3
                int32x4_t m0 = vreinterpretq_s32_u32(vshlq_u32(p_vec, sh0));
                uint32x4_t mask0 = vreinterpretq_u32_s32(vshrq_n_s32(m0, 31));
                c0_0 = vaddq_f32(c0_0, vreinterpretq_f32_u32(vandq_u32(a0, mask0)));
                c1_0 = vaddq_f32(c1_0, vreinterpretq_f32_u32(vandq_u32(a1, mask0)));

                // Bit 4..7
                int32x4_t m1 = vreinterpretq_s32_u32(vshlq_u32(p_vec, sh1));
                uint32x4_t mask1 = vreinterpretq_u32_s32(vshrq_n_s32(m1, 31));
                c0_1 = vaddq_f32(c0_1, vreinterpretq_f32_u32(vandq_u32(a0, mask1)));
                c1_1 = vaddq_f32(c1_1, vreinterpretq_f32_u32(vandq_u32(a1, mask1)));

                // Bit 8..11
                int32x4_t m2 = vreinterpretq_s32_u32(vshlq_u32(p_vec, sh2));
                uint32x4_t mask2 = vreinterpretq_u32_s32(vshrq_n_s32(m2, 31));
                c0_2 = vaddq_f32(c0_2, vreinterpretq_f32_u32(vandq_u32(a0, mask2)));
                c1_2 = vaddq_f32(c1_2, vreinterpretq_f32_u32(vandq_u32(a1, mask2)));

                // Bit 12..15
                int32x4_t m3 = vreinterpretq_s32_u32(vshlq_u32(p_vec, sh3));
                uint32x4_t mask3 = vreinterpretq_u32_s32(vshrq_n_s32(m3, 31));
                c0_3 = vaddq_f32(c0_3, vreinterpretq_f32_u32(vandq_u32(a0, mask3)));
                c1_3 = vaddq_f32(c1_3, vreinterpretq_f32_u32(vandq_u32(a1, mask3)));

                // Bit 16..19
                int32x4_t m4 = vreinterpretq_s32_u32(vshlq_u32(p_vec, sh4));
                uint32x4_t mask4 = vreinterpretq_u32_s32(vshrq_n_s32(m4, 31));
                c0_4 = vaddq_f32(c0_4, vreinterpretq_f32_u32(vandq_u32(a0, mask4)));
                c1_4 = vaddq_f32(c1_4, vreinterpretq_f32_u32(vandq_u32(a1, mask4)));

                // Bit 20..23
                int32x4_t m5 = vreinterpretq_s32_u32(vshlq_u32(p_vec, sh5));
                uint32x4_t mask5 = vreinterpretq_u32_s32(vshrq_n_s32(m5, 31));
                c0_5 = vaddq_f32(c0_5, vreinterpretq_f32_u32(vandq_u32(a0, mask5)));
                c1_5 = vaddq_f32(c1_5, vreinterpretq_f32_u32(vandq_u32(a1, mask5)));

                // Bit 24..27
                int32x4_t m6 = vreinterpretq_s32_u32(vshlq_u32(p_vec, sh6));
                uint32x4_t mask6 = vreinterpretq_u32_s32(vshrq_n_s32(m6, 31));
                c0_6 = vaddq_f32(c0_6, vreinterpretq_f32_u32(vandq_u32(a0, mask6)));
                c1_6 = vaddq_f32(c1_6, vreinterpretq_f32_u32(vandq_u32(a1, mask6)));

                // Bit 28..31
                int32x4_t m7 = vreinterpretq_s32_u32(vshlq_u32(p_vec, sh7));
                uint32x4_t mask7 = vreinterpretq_u32_s32(vshrq_n_s32(m7, 31));
                c0_7 = vaddq_f32(c0_7, vreinterpretq_f32_u32(vandq_u32(a0, mask7)));
                c1_7 = vaddq_f32(c1_7, vreinterpretq_f32_u32(vandq_u32(a1, mask7)));
            }

            float32x4_t rsum0 = vdupq_n_f32(row_sums[i + 0]);
            float32x4_t rsum1 = vdupq_n_f32(row_sums[i + 1]);

            float32x4_t two = vdupq_n_f32(2.0f);

            c0_0 = vmlsq_f32(rsum0, c0_0, two); // wait, c = 2*C - rowsum. So vmlsq is wrong. 
            // C = c * 2.0 - rsum
            c0_0 = vmlaq_f32(vnegq_f32(rsum0), c0_0, two);
            c0_1 = vmlaq_f32(vnegq_f32(rsum0), c0_1, two);
            c0_2 = vmlaq_f32(vnegq_f32(rsum0), c0_2, two);
            c0_3 = vmlaq_f32(vnegq_f32(rsum0), c0_3, two);
            c0_4 = vmlaq_f32(vnegq_f32(rsum0), c0_4, two);
            c0_5 = vmlaq_f32(vnegq_f32(rsum0), c0_5, two);
            c0_6 = vmlaq_f32(vnegq_f32(rsum0), c0_6, two);
            c0_7 = vmlaq_f32(vnegq_f32(rsum0), c0_7, two);

            c1_0 = vmlaq_f32(vnegq_f32(rsum1), c1_0, two);
            c1_1 = vmlaq_f32(vnegq_f32(rsum1), c1_1, two);
            c1_2 = vmlaq_f32(vnegq_f32(rsum1), c1_2, two);
            c1_3 = vmlaq_f32(vnegq_f32(rsum1), c1_3, two);
            c1_4 = vmlaq_f32(vnegq_f32(rsum1), c1_4, two);
            c1_5 = vmlaq_f32(vnegq_f32(rsum1), c1_5, two);
            c1_6 = vmlaq_f32(vnegq_f32(rsum1), c1_6, two);
            c1_7 = vmlaq_f32(vnegq_f32(rsum1), c1_7, two);

            float* C_ptr0 = &C[(i + 0) * K + j_int * 32];
            vst1q_f32(C_ptr0 + 0, c0_0);
            vst1q_f32(C_ptr0 + 4, c0_1);
            vst1q_f32(C_ptr0 + 8, c0_2);
            vst1q_f32(C_ptr0 + 12, c0_3);
            vst1q_f32(C_ptr0 + 16, c0_4);
            vst1q_f32(C_ptr0 + 20, c0_5);
            vst1q_f32(C_ptr0 + 24, c0_6);
            vst1q_f32(C_ptr0 + 28, c0_7);

            float* C_ptr1 = &C[(i + 1) * K + j_int * 32];
            vst1q_f32(C_ptr1 + 0, c1_0);
            vst1q_f32(C_ptr1 + 4, c1_1);
            vst1q_f32(C_ptr1 + 8, c1_2);
            vst1q_f32(C_ptr1 + 12, c1_3);
            vst1q_f32(C_ptr1 + 16, c1_4);
            vst1q_f32(C_ptr1 + 20, c1_5);
            vst1q_f32(C_ptr1 + 24, c1_6);
            vst1q_f32(C_ptr1 + 28, c1_7);
        }
    }

    for (; i < M; ++i) {
        for (size_t j_int = 0; j_int < K_ints; ++j_int) {
            float c[32] = {0};
            for (size_t p = 0; p < K; ++p) {
                float a = A[i * K + p];
                uint32_t packed = B[p * K_ints + j_int];
                for (int b = 0; b < 32; ++b) {
                    uint32_t bit = (packed >> b) & 1;
                    float fbit = bit;
                    c[b] += fbit * a;
                }
            }
            for (int b = 0; b < 32; ++b) {
                C[i * K + j_int * 32 + b] = 2.0f * c[b] - row_sums[i];
            }
        }
    }
}
