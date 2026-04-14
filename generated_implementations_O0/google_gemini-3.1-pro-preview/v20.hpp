#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            row_sum += A[i * K + p];
        }

        for (size_t j_int = 0; j_int < K_ints; ++j_int) {
            float32x4_t c0 = vdupq_n_f32(0);
            float32x4_t c1 = vdupq_n_f32(0);
            float32x4_t c2 = vdupq_n_f32(0);
            float32x4_t c3 = vdupq_n_f32(0);
            float32x4_t c4 = vdupq_n_f32(0);
            float32x4_t c5 = vdupq_n_f32(0);
            float32x4_t c6 = vdupq_n_f32(0);
            float32x4_t c7 = vdupq_n_f32(0);

            for (size_t p = 0; p < K; ++p) {
                float a = A[i * K + p];
                float32x4_t a_vec = vdupq_n_f32(a);

                uint32_t packed = B[p * K_ints + j_int];
                uint32x4_t p_vec = vdupq_n_u32(packed);

                // Need 32 shifts to extract all bits, multiply by a, and add.
                // We can do this efficiently using standard NEON instructions.
                
                int32x4_t sh0 = {0, -1, -2, -3};
                uint32x4_t m0 = vshlq_u32(p_vec, sh0);
                m0 = vandq_u32(m0, vdupq_n_u32(1));
                c0 = vmlaq_f32(c0, a_vec, vcvtq_f32_u32(m0));

                int32x4_t sh1 = {-4, -5, -6, -7};
                uint32x4_t m1 = vshlq_u32(p_vec, sh1);
                m1 = vandq_u32(m1, vdupq_n_u32(1));
                c1 = vmlaq_f32(c1, a_vec, vcvtq_f32_u32(m1));

                int32x4_t sh2 = {-8, -9, -10, -11};
                uint32x4_t m2 = vshlq_u32(p_vec, sh2);
                m2 = vandq_u32(m2, vdupq_n_u32(1));
                c2 = vmlaq_f32(c2, a_vec, vcvtq_f32_u32(m2));

                int32x4_t sh3 = {-12, -13, -14, -15};
                uint32x4_t m3 = vshlq_u32(p_vec, sh3);
                m3 = vandq_u32(m3, vdupq_n_u32(1));
                c3 = vmlaq_f32(c3, a_vec, vcvtq_f32_u32(m3));

                int32x4_t sh4 = {-16, -17, -18, -19};
                uint32x4_t m4 = vshlq_u32(p_vec, sh4);
                m4 = vandq_u32(m4, vdupq_n_u32(1));
                c4 = vmlaq_f32(c4, a_vec, vcvtq_f32_u32(m4));

                int32x4_t sh5 = {-20, -21, -22, -23};
                uint32x4_t m5 = vshlq_u32(p_vec, sh5);
                m5 = vandq_u32(m5, vdupq_n_u32(1));
                c5 = vmlaq_f32(c5, a_vec, vcvtq_f32_u32(m5));

                int32x4_t sh6 = {-24, -25, -26, -27};
                uint32x4_t m6 = vshlq_u32(p_vec, sh6);
                m6 = vandq_u32(m6, vdupq_n_u32(1));
                c6 = vmlaq_f32(c6, a_vec, vcvtq_f32_u32(m6));

                int32x4_t sh7 = {-28, -29, -30, -31};
                uint32x4_t m7 = vshlq_u32(p_vec, sh7);
                m7 = vandq_u32(m7, vdupq_n_u32(1));
                c7 = vmlaq_f32(c7, a_vec, vcvtq_f32_u32(m7));
            }

            float32x4_t r_vec = vdupq_n_f32(row_sum);
            float32x4_t two = vdupq_n_f32(2.0f);

            c0 = vmlsq_f32(vnegq_f32(r_vec), c0, vnegq_f32(two));
            c1 = vmlsq_f32(vnegq_f32(r_vec), c1, vnegq_f32(two));
            c2 = vmlsq_f32(vnegq_f32(r_vec), c2, vnegq_f32(two));
            c3 = vmlsq_f32(vnegq_f32(r_vec), c3, vnegq_f32(two));
            c4 = vmlsq_f32(vnegq_f32(r_vec), c4, vnegq_f32(two));
            c5 = vmlsq_f32(vnegq_f32(r_vec), c5, vnegq_f32(two));
            c6 = vmlsq_f32(vnegq_f32(r_vec), c6, vnegq_f32(two));
            c7 = vmlsq_f32(vnegq_f32(r_vec), c7, vnegq_f32(two));

            vst1q_f32(&C[i * K + j_int * 32 + 0 ], c0);
            vst1q_f32(&C[i * K + j_int * 32 + 4 ], c1);
            vst1q_f32(&C[i * K + j_int * 32 + 8 ], c2);
            vst1q_f32(&C[i * K + j_int * 32 + 12], c3);
            vst1q_f32(&C[i * K + j_int * 32 + 16], c4);
            vst1q_f32(&C[i * K + j_int * 32 + 20], c5);
            vst1q_f32(&C[i * K + j_int * 32 + 24], c6);
            vst1q_f32(&C[i * K + j_int * 32 + 28], c7);
        }
    }
}
