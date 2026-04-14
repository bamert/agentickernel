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
    int32x4_t sh5 = {11, 10, 9, 8};
    int32x4_t sh6 = {7,  6,  5,  4};
    int32x4_t sh7 = {3,  2,  1,  0};
    uint32x4_t sign_mask = vdupq_n_u32(0x80000000);

    for (size_t i = 0; i < M * K; ++i) {
        C[i] = 0.0f;
    }

    for (size_t i = 0; i < M; ++i) {
        float* C_row = &C[i * K];
        const float* A_row = &A[i * K];

        size_t p = 0;
        // Unroll p by 4
        for (; p + 3 < K; p += 4) {
            float a0 = A_row[p + 0];
            float a1 = A_row[p + 1];
            float a2 = A_row[p + 2];
            float a3 = A_row[p + 3];

            uint32x4_t av0 = vreinterpretq_u32_f32(vdupq_n_f32(a0));
            uint32x4_t av1 = vreinterpretq_u32_f32(vdupq_n_f32(a1));
            uint32x4_t av2 = vreinterpretq_u32_f32(vdupq_n_f32(a2));
            uint32x4_t av3 = vreinterpretq_u32_f32(vdupq_n_f32(a3));

            const uint32_t* B_row0 = &B[(p + 0) * K_ints];
            const uint32_t* B_row1 = &B[(p + 1) * K_ints];
            const uint32_t* B_row2 = &B[(p + 2) * K_ints];
            const uint32_t* B_row3 = &B[(p + 3) * K_ints];

            for (size_t j = 0; j < K_ints; ++j) {
                float* C_ptr = C_row + j * 32;

                uint32x4_t p_vec0 = vdupq_n_u32(~B_row0[j]);
                uint32x4_t p_vec1 = vdupq_n_u32(~B_row1[j]);
                uint32x4_t p_vec2 = vdupq_n_u32(~B_row2[j]);
                uint32x4_t p_vec3 = vdupq_n_u32(~B_row3[j]);

                // We load 8 vectors of C and add for all 4 bits
                // To avoid register pressure, block by 4 vectors at a time (16 floats, half a uint32)
                
                // First half (sh0 to sh3)
                float32x4_t c0 = vld1q_f32(C_ptr + 0);
                float32x4_t c1 = vld1q_f32(C_ptr + 4);
                float32x4_t c2 = vld1q_f32(C_ptr + 8);
                float32x4_t c3 = vld1q_f32(C_ptr + 12);

                // p=0
                c0 = vaddq_f32(c0, vreinterpretq_f32_u32(veorq_u32(av0, vandq_u32(vshlq_u32(p_vec0, sh0), sign_mask))));
                c1 = vaddq_f32(c1, vreinterpretq_f32_u32(veorq_u32(av0, vandq_u32(vshlq_u32(p_vec0, sh1), sign_mask))));
                c2 = vaddq_f32(c2, vreinterpretq_f32_u32(veorq_u32(av0, vandq_u32(vshlq_u32(p_vec0, sh2), sign_mask))));
                c3 = vaddq_f32(c3, vreinterpretq_f32_u32(veorq_u32(av0, vandq_u32(vshlq_u32(p_vec0, sh3), sign_mask))));

                // p=1
                c0 = vaddq_f32(c0, vreinterpretq_f32_u32(veorq_u32(av1, vandq_u32(vshlq_u32(p_vec1, sh0), sign_mask))));
                c1 = vaddq_f32(c1, vreinterpretq_f32_u32(veorq_u32(av1, vandq_u32(vshlq_u32(p_vec1, sh1), sign_mask))));
                c2 = vaddq_f32(c2, vreinterpretq_f32_u32(veorq_u32(av1, vandq_u32(vshlq_u32(p_vec1, sh2), sign_mask))));
                c3 = vaddq_f32(c3, vreinterpretq_f32_u32(veorq_u32(av1, vandq_u32(vshlq_u32(p_vec1, sh3), sign_mask))));

                // p=2
                c0 = vaddq_f32(c0, vreinterpretq_f32_u32(veorq_u32(av2, vandq_u32(vshlq_u32(p_vec2, sh0), sign_mask))));
                c1 = vaddq_f32(c1, vreinterpretq_f32_u32(veorq_u32(av2, vandq_u32(vshlq_u32(p_vec2, sh1), sign_mask))));
                c2 = vaddq_f32(c2, vreinterpretq_f32_u32(veorq_u32(av2, vandq_u32(vshlq_u32(p_vec2, sh2), sign_mask))));
                c3 = vaddq_f32(c3, vreinterpretq_f32_u32(veorq_u32(av2, vandq_u32(vshlq_u32(p_vec2, sh3), sign_mask))));

                // p=3
                c0 = vaddq_f32(c0, vreinterpretq_f32_u32(veorq_u32(av3, vandq_u32(vshlq_u32(p_vec3, sh0), sign_mask))));
                c1 = vaddq_f32(c1, vreinterpretq_f32_u32(veorq_u32(av3, vandq_u32(vshlq_u32(p_vec3, sh1), sign_mask))));
                c2 = vaddq_f32(c2, vreinterpretq_f32_u32(veorq_u32(av3, vandq_u32(vshlq_u32(p_vec3, sh2), sign_mask))));
                c3 = vaddq_f32(c3, vreinterpretq_f32_u32(veorq_u32(av3, vandq_u32(vshlq_u32(p_vec3, sh3), sign_mask))));

                vst1q_f32(C_ptr + 0, c0);
                vst1q_f32(C_ptr + 4, c1);
                vst1q_f32(C_ptr + 8, c2);
                vst1q_f32(C_ptr + 12, c3);

                // Second half (sh4 to sh7)
                float32x4_t c4 = vld1q_f32(C_ptr + 16);
                float32x4_t c5 = vld1q_f32(C_ptr + 20);
                float32x4_t c6 = vld1q_f32(C_ptr + 24);
                float32x4_t c7 = vld1q_f32(C_ptr + 28);

                // p=0
                c4 = vaddq_f32(c4, vreinterpretq_f32_u32(veorq_u32(av0, vandq_u32(vshlq_u32(p_vec0, sh4), sign_mask))));
                c5 = vaddq_f32(c5, vreinterpretq_f32_u32(veorq_u32(av0, vandq_u32(vshlq_u32(p_vec0, sh5), sign_mask))));
                c6 = vaddq_f32(c6, vreinterpretq_f32_u32(veorq_u32(av0, vandq_u32(vshlq_u32(p_vec0, sh6), sign_mask))));
                c7 = vaddq_f32(c7, vreinterpretq_f32_u32(veorq_u32(av0, vandq_u32(vshlq_u32(p_vec0, sh7), sign_mask))));

                // p=1
                c4 = vaddq_f32(c4, vreinterpretq_f32_u32(veorq_u32(av1, vandq_u32(vshlq_u32(p_vec1, sh4), sign_mask))));
                c5 = vaddq_f32(c5, vreinterpretq_f32_u32(veorq_u32(av1, vandq_u32(vshlq_u32(p_vec1, sh5), sign_mask))));
                c6 = vaddq_f32(c6, vreinterpretq_f32_u32(veorq_u32(av1, vandq_u32(vshlq_u32(p_vec1, sh6), sign_mask))));
                c7 = vaddq_f32(c7, vreinterpretq_f32_u32(veorq_u32(av1, vandq_u32(vshlq_u32(p_vec1, sh7), sign_mask))));

                // p=2
                c4 = vaddq_f32(c4, vreinterpretq_f32_u32(veorq_u32(av2, vandq_u32(vshlq_u32(p_vec2, sh4), sign_mask))));
                c5 = vaddq_f32(c5, vreinterpretq_f32_u32(veorq_u32(av2, vandq_u32(vshlq_u32(p_vec2, sh5), sign_mask))));
                c6 = vaddq_f32(c6, vreinterpretq_f32_u32(veorq_u32(av2, vandq_u32(vshlq_u32(p_vec2, sh6), sign_mask))));
                c7 = vaddq_f32(c7, vreinterpretq_f32_u32(veorq_u32(av2, vandq_u32(vshlq_u32(p_vec2, sh7), sign_mask))));

                // p=3
                c4 = vaddq_f32(c4, vreinterpretq_f32_u32(veorq_u32(av3, vandq_u32(vshlq_u32(p_vec3, sh4), sign_mask))));
                c5 = vaddq_f32(c5, vreinterpretq_f32_u32(veorq_u32(av3, vandq_u32(vshlq_u32(p_vec3, sh5), sign_mask))));
                c6 = vaddq_f32(c6, vreinterpretq_f32_u32(veorq_u32(av3, vandq_u32(vshlq_u32(p_vec3, sh6), sign_mask))));
                c7 = vaddq_f32(c7, vreinterpretq_f32_u32(veorq_u32(av3, vandq_u32(vshlq_u32(p_vec3, sh7), sign_mask))));

                vst1q_f32(C_ptr + 16, c4);
                vst1q_f32(C_ptr + 20, c5);
                vst1q_f32(C_ptr + 24, c6);
                vst1q_f32(C_ptr + 28, c7);
            }
        }
        
        for (; p < K; ++p) {
            float a = A_row[p];
            uint32x4_t a_vec = vreinterpretq_u32_f32(vdupq_n_f32(a));
            const uint32_t* B_row = &B[p * K_ints];

            for (size_t j = 0; j < K_ints; ++j) {
                float* C_ptr = C_row + j * 32;
                uint32x4_t p_vec = vdupq_n_u32(~B_row[j]);
                
                float32x4_t c0 = vld1q_f32(C_ptr + 0);
                float32x4_t c1 = vld1q_f32(C_ptr + 4);
                float32x4_t c2 = vld1q_f32(C_ptr + 8);
                float32x4_t c3 = vld1q_f32(C_ptr + 12);

                c0 = vaddq_f32(c0, vreinterpretq_f32_u32(veorq_u32(a_vec, vandq_u32(vshlq_u32(p_vec, sh0), sign_mask))));
                c1 = vaddq_f32(c1, vreinterpretq_f32_u32(veorq_u32(a_vec, vandq_u32(vshlq_u32(p_vec, sh1), sign_mask))));
                c2 = vaddq_f32(c2, vreinterpretq_f32_u32(veorq_u32(a_vec, vandq_u32(vshlq_u32(p_vec, sh2), sign_mask))));
                c3 = vaddq_f32(c3, vreinterpretq_f32_u32(veorq_u32(a_vec, vandq_u32(vshlq_u32(p_vec, sh3), sign_mask))));

                vst1q_f32(C_ptr + 0, c0);
                vst1q_f32(C_ptr + 4, c1);
                vst1q_f32(C_ptr + 8, c2);
                vst1q_f32(C_ptr + 12, c3);

                float32x4_t c4 = vld1q_f32(C_ptr + 16);
                float32x4_t c5 = vld1q_f32(C_ptr + 20);
                float32x4_t c6 = vld1q_f32(C_ptr + 24);
                float32x4_t c7 = vld1q_f32(C_ptr + 28);

                c4 = vaddq_f32(c4, vreinterpretq_f32_u32(veorq_u32(a_vec, vandq_u32(vshlq_u32(p_vec, sh4), sign_mask))));
                c5 = vaddq_f32(c5, vreinterpretq_f32_u32(veorq_u32(a_vec, vandq_u32(vshlq_u32(p_vec, sh5), sign_mask))));
                c6 = vaddq_f32(c6, vreinterpretq_f32_u32(veorq_u32(a_vec, vandq_u32(vshlq_u32(p_vec, sh6), sign_mask))));
                c7 = vaddq_f32(c7, vreinterpretq_f32_u32(veorq_u32(a_vec, vandq_u32(vshlq_u32(p_vec, sh7), sign_mask))));

                vst1q_f32(C_ptr + 16, c4);
                vst1q_f32(C_ptr + 20, c5);
                vst1q_f32(C_ptr + 24, c6);
                vst1q_f32(C_ptr + 28, c7);
            }
        }
    }
}
