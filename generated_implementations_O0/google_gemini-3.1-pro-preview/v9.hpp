#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    int32x4_t sh0 = {31, 30, 29, 28};
    int32x4_t sh1 = {27, 26, 25, 24};
    int32x4_t sh2 = {23, 22, 21, 20};
    int32x4_t sh3 = {19, 18, 17, 16};
    int32x4_t sh4 = {15, 14, 13, 12};
    int32x4_t sh5 = {11, 10,  9,  8};
    int32x4_t sh6 = { 7,  6,  5,  4};
    int32x4_t sh7 = { 3,  2,  1,  0};
    uint32x4_t sign_mask = vdupq_n_u32(0x80000000);

    size_t i = 0;
    for (; i + 1 < M; i += 2) {
        for (size_t j_int = 0; j_int < K_ints; ++j_int) {
            float32x4_t c0_0 = vdupq_n_f32(0); float32x4_t c0_1 = vdupq_n_f32(0);
            float32x4_t c0_2 = vdupq_n_f32(0); float32x4_t c0_3 = vdupq_n_f32(0);
            float32x4_t c0_4 = vdupq_n_f32(0); float32x4_t c0_5 = vdupq_n_f32(0);
            float32x4_t c0_6 = vdupq_n_f32(0); float32x4_t c0_7 = vdupq_n_f32(0);

            float32x4_t c1_0 = vdupq_n_f32(0); float32x4_t c1_1 = vdupq_n_f32(0);
            float32x4_t c1_2 = vdupq_n_f32(0); float32x4_t c1_3 = vdupq_n_f32(0);
            float32x4_t c1_4 = vdupq_n_f32(0); float32x4_t c1_5 = vdupq_n_f32(0);
            float32x4_t c1_6 = vdupq_n_f32(0); float32x4_t c1_7 = vdupq_n_f32(0);

            for (size_t p = 0; p < K; ++p) {
                float a0 = A[(i + 0) * K + p];
                float a1 = A[(i + 1) * K + p];
                uint32x4_t a0_vec = vreinterpretq_u32_f32(vdupq_n_f32(a0));
                uint32x4_t a1_vec = vreinterpretq_u32_f32(vdupq_n_f32(a1));

                uint32_t packed = B[p * K_ints + j_int];
                uint32x4_t p_vec = vdupq_n_u32(~packed);

                // Inline macro strictly updates the registers
                #define PROCESS_CHUNK(id, sh_vec) \
                { \
                    uint32x4_t shifted = vandq_u32(vshlq_u32(p_vec, sh_vec), sign_mask); \
                    c0_##id = vaddq_f32(c0_##id, vreinterpretq_f32_u32(veorq_u32(a0_vec, shifted))); \
                    c1_##id = vaddq_f32(c1_##id, vreinterpretq_f32_u32(veorq_u32(a1_vec, shifted))); \
                }

                PROCESS_CHUNK(0, sh0)
                PROCESS_CHUNK(1, sh1)
                PROCESS_CHUNK(2, sh2)
                PROCESS_CHUNK(3, sh3)
                PROCESS_CHUNK(4, sh4)
                PROCESS_CHUNK(5, sh5)
                PROCESS_CHUNK(6, sh6)
                PROCESS_CHUNK(7, sh7)
                
                #undef PROCESS_CHUNK
            }

            vst1q_f32(&C[(i + 0) * K + j_int * 32 + 0 ], c0_0);
            vst1q_f32(&C[(i + 0) * K + j_int * 32 + 4 ], c0_1);
            vst1q_f32(&C[(i + 0) * K + j_int * 32 + 8 ], c0_2);
            vst1q_f32(&C[(i + 0) * K + j_int * 32 + 12], c0_3);
            vst1q_f32(&C[(i + 0) * K + j_int * 32 + 16], c0_4);
            vst1q_f32(&C[(i + 0) * K + j_int * 32 + 20], c0_5);
            vst1q_f32(&C[(i + 0) * K + j_int * 32 + 24], c0_6);
            vst1q_f32(&C[(i + 0) * K + j_int * 32 + 28], c0_7);

            vst1q_f32(&C[(i + 1) * K + j_int * 32 + 0 ], c1_0);
            vst1q_f32(&C[(i + 1) * K + j_int * 32 + 4 ], c1_1);
            vst1q_f32(&C[(i + 1) * K + j_int * 32 + 8 ], c1_2);
            vst1q_f32(&C[(i + 1) * K + j_int * 32 + 12], c1_3);
            vst1q_f32(&C[(i + 1) * K + j_int * 32 + 16], c1_4);
            vst1q_f32(&C[(i + 1) * K + j_int * 32 + 20], c1_5);
            vst1q_f32(&C[(i + 1) * K + j_int * 32 + 24], c1_6);
            vst1q_f32(&C[(i + 1) * K + j_int * 32 + 28], c1_7);
        }
    }

    for (; i < M; ++i) {
        for (size_t j_int = 0; j_int < K_ints; ++j_int) {
            float32x4_t c0_0 = vdupq_n_f32(0); float32x4_t c0_1 = vdupq_n_f32(0);
            float32x4_t c0_2 = vdupq_n_f32(0); float32x4_t c0_3 = vdupq_n_f32(0);
            float32x4_t c0_4 = vdupq_n_f32(0); float32x4_t c0_5 = vdupq_n_f32(0);
            float32x4_t c0_6 = vdupq_n_f32(0); float32x4_t c0_7 = vdupq_n_f32(0);

            for (size_t p = 0; p < K; ++p) {
                float a0 = A[i * K + p];
                uint32x4_t a0_vec = vreinterpretq_u32_f32(vdupq_n_f32(a0));
                
                uint32_t packed = B[p * K_ints + j_int];
                uint32x4_t p_vec = vdupq_n_u32(~packed);

                #define PROCESS_CHUNK(id, sh_vec) \
                { \
                    uint32x4_t shifted = vandq_u32(vshlq_u32(p_vec, sh_vec), sign_mask); \
                    c0_##id = vaddq_f32(c0_##id, vreinterpretq_f32_u32(veorq_u32(a0_vec, shifted))); \
                }

                PROCESS_CHUNK(0, sh0)
                PROCESS_CHUNK(1, sh1)
                PROCESS_CHUNK(2, sh2)
                PROCESS_CHUNK(3, sh3)
                PROCESS_CHUNK(4, sh4)
                PROCESS_CHUNK(5, sh5)
                PROCESS_CHUNK(6, sh6)
                PROCESS_CHUNK(7, sh7)
                
                #undef PROCESS_CHUNK
            }

            vst1q_f32(&C[i * K + j_int * 32 + 0 ], c0_0);
            vst1q_f32(&C[i * K + j_int * 32 + 4 ], c0_1);
            vst1q_f32(&C[i * K + j_int * 32 + 8 ], c0_2);
            vst1q_f32(&C[i * K + j_int * 32 + 12], c0_3);
            vst1q_f32(&C[i * K + j_int * 32 + 16], c0_4);
            vst1q_f32(&C[i * K + j_int * 32 + 20], c0_5);
            vst1q_f32(&C[i * K + j_int * 32 + 24], c0_6);
            vst1q_f32(&C[i * K + j_int * 32 + 28], c0_7);
        }
    }
}
