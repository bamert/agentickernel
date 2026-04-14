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
        for (size_t p = 0; p < K; ++p) sum += A[i * K + p];
        row_sums[i] = sum;
    }

    size_t i = 0;
    for (; i + 3 < M; i += 4) {
        for (size_t j_int = 0; j_int < K_ints; ++j_int) {
            float32x4_t c0[8], c1[8], c2[8], c3[8];
            for (int k = 0; k < 8; ++k) {
                c0[k] = vdupq_n_f32(0.0f);
                c1[k] = vdupq_n_f32(0.0f);
                c2[k] = vdupq_n_f32(0.0f);
                c3[k] = vdupq_n_f32(0.0f);
            }

            for (size_t p = 0; p < K; ++p) {
                float32x4_t a_vec = {
                    A[(i + 0) * K + p],
                    A[(i + 1) * K + p],
                    A[(i + 2) * K + p],
                    A[(i + 3) * K + p]
                };

                uint32_t pk = B[p * K_ints + j_int];
                uint32x4_t pk_vec = vdupq_n_u32(pk);
                
                int32x4_t sh0 = {0, -1, -2, -3};
                int32x4_t sh1 = {-4, -5, -6, -7};
                int32x4_t sh2 = {-8, -9, -10, -11};
                int32x4_t sh3 = {-12, -13, -14, -15};
                int32x4_t sh4 = {-16, -17, -18, -19};
                int32x4_t sh5 = {-20, -21, -22, -23};
                int32x4_t sh6 = {-24, -25, -26, -27};
                int32x4_t sh7 = {-28, -29, -30, -31};

                uint32x4_t m0 = vandq_u32(vshlq_u32(pk_vec, sh0), vdupq_n_u32(1));
                uint32x4_t m1 = vandq_u32(vshlq_u32(pk_vec, sh1), vdupq_n_u32(1));
                uint32x4_t m2 = vandq_u32(vshlq_u32(pk_vec, sh2), vdupq_n_u32(1));
                uint32x4_t m3 = vandq_u32(vshlq_u32(pk_vec, sh3), vdupq_n_u32(1));
                uint32x4_t m4 = vandq_u32(vshlq_u32(pk_vec, sh4), vdupq_n_u32(1));
                uint32x4_t m5 = vandq_u32(vshlq_u32(pk_vec, sh5), vdupq_n_u32(1));
                uint32x4_t m6 = vandq_u32(vshlq_u32(pk_vec, sh6), vdupq_n_u32(1));
                uint32x4_t m7 = vandq_u32(vshlq_u32(pk_vec, sh7), vdupq_n_u32(1));

                float32x4_t f0 = vcvtq_f32_u32(m0);
                float32x4_t f1 = vcvtq_f32_u32(m1);
                float32x4_t f2 = vcvtq_f32_u32(m2);
                float32x4_t f3 = vcvtq_f32_u32(m3);
                float32x4_t f4 = vcvtq_f32_u32(m4);
                float32x4_t f5 = vcvtq_f32_u32(m5);
                float32x4_t f6 = vcvtq_f32_u32(m6);
                float32x4_t f7 = vcvtq_f32_u32(m7);

                c0[0] = vmlaq_n_f32(c0[0], f0, vgetq_lane_f32(a_vec, 0));
                c1[0] = vmlaq_n_f32(c1[0], f0, vgetq_lane_f32(a_vec, 1));
                c2[0] = vmlaq_n_f32(c2[0], f0, vgetq_lane_f32(a_vec, 2));
                c3[0] = vmlaq_n_f32(c3[0], f0, vgetq_lane_f32(a_vec, 3));

                c0[1] = vmlaq_n_f32(c0[1], f1, vgetq_lane_f32(a_vec, 0));
                c1[1] = vmlaq_n_f32(c1[1], f1, vgetq_lane_f32(a_vec, 1));
                c2[1] = vmlaq_n_f32(c2[1], f1, vgetq_lane_f32(a_vec, 2));
                c3[1] = vmlaq_n_f32(c3[1], f1, vgetq_lane_f32(a_vec, 3));

                c0[2] = vmlaq_n_f32(c0[2], f2, vgetq_lane_f32(a_vec, 0));
                c1[2] = vmlaq_n_f32(c1[2], f2, vgetq_lane_f32(a_vec, 1));
                c2[2] = vmlaq_n_f32(c2[2], f2, vgetq_lane_f32(a_vec, 2));
                c3[2] = vmlaq_n_f32(c3[2], f2, vgetq_lane_f32(a_vec, 3));

                c0[3] = vmlaq_n_f32(c0[3], f3, vgetq_lane_f32(a_vec, 0));
                c1[3] = vmlaq_n_f32(c1[3], f3, vgetq_lane_f32(a_vec, 1));
                c2[3] = vmlaq_n_f32(c2[3], f3, vgetq_lane_f32(a_vec, 2));
                c3[3] = vmlaq_n_f32(c3[3], f3, vgetq_lane_f32(a_vec, 3));

                c0[4] = vmlaq_n_f32(c0[4], f4, vgetq_lane_f32(a_vec, 0));
                c1[4] = vmlaq_n_f32(c1[4], f4, vgetq_lane_f32(a_vec, 1));
                c2[4] = vmlaq_n_f32(c2[4], f4, vgetq_lane_f32(a_vec, 2));
                c3[4] = vmlaq_n_f32(c3[4], f4, vgetq_lane_f32(a_vec, 3));

                c0[5] = vmlaq_n_f32(c0[5], f5, vgetq_lane_f32(a_vec, 0));
                c1[5] = vmlaq_n_f32(c1[5], f5, vgetq_lane_f32(a_vec, 1));
                c2[5] = vmlaq_n_f32(c2[5], f5, vgetq_lane_f32(a_vec, 2));
                c3[5] = vmlaq_n_f32(c3[5], f5, vgetq_lane_f32(a_vec, 3));

                c0[6] = vmlaq_n_f32(c0[6], f6, vgetq_lane_f32(a_vec, 0));
                c1[6] = vmlaq_n_f32(c1[6], f6, vgetq_lane_f32(a_vec, 1));
                c2[6] = vmlaq_n_f32(c2[6], f6, vgetq_lane_f32(a_vec, 2));
                c3[6] = vmlaq_n_f32(c3[6], f6, vgetq_lane_f32(a_vec, 3));

                c0[7] = vmlaq_n_f32(c0[7], f7, vgetq_lane_f32(a_vec, 0));
                c1[7] = vmlaq_n_f32(c1[7], f7, vgetq_lane_f32(a_vec, 1));
                c2[7] = vmlaq_n_f32(c2[7], f7, vgetq_lane_f32(a_vec, 2));
                c3[7] = vmlaq_n_f32(c3[7], f7, vgetq_lane_f32(a_vec, 3));
            }

            float32x4_t rs0 = vdupq_n_f32(row_sums[i + 0]);
            float32x4_t rs1 = vdupq_n_f32(row_sums[i + 1]);
            float32x4_t rs2 = vdupq_n_f32(row_sums[i + 2]);
            float32x4_t rs3 = vdupq_n_f32(row_sums[i + 3]);
            float32x4_t two = vdupq_n_f32(2.0f);

            for (int k = 0; k < 8; ++k) {
                c0[k] = vmlsq_f32(vnegq_f32(rs0), c0[k], vnegq_f32(two));
                c1[k] = vmlsq_f32(vnegq_f32(rs1), c1[k], vnegq_f32(two));
                c2[k] = vmlsq_f32(vnegq_f32(rs2), c2[k], vnegq_f32(two));
                c3[k] = vmlsq_f32(vnegq_f32(rs3), c3[k], vnegq_f32(two));

                vst1q_f32(&C[(i + 0) * K + j_int * 32 + k * 4], c0[k]);
                vst1q_f32(&C[(i + 1) * K + j_int * 32 + k * 4], c1[k]);
                vst1q_f32(&C[(i + 2) * K + j_int * 32 + k * 4], c2[k]);
                vst1q_f32(&C[(i + 3) * K + j_int * 32 + k * 4], c3[k]);
            }
        }
    }
    
    for (; i < M; ++i) {
        for (size_t j_int = 0; j_int < K_ints; ++j_int) {
            float c[32] = {0};
            for (size_t p = 0; p < K; ++p) {
                float a = A[i * K + p];
                uint32_t packed = B[p * K_ints + j_int];
                for (int b = 0; b < 32; ++b) {
                    float fbit = (packed & 1);
                    packed >>= 1;
                    c[b] += fbit * a;
                }
            }
            float rs = row_sums[i];
            float* C_ptr = &C[i * K + j_int * 32];
            for (int b = 0; b < 32; ++b) {
                C_ptr[b] = 2.0f * c[b] - rs;
            }
        }
    }
}
