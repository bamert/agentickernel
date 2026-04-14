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
                c0[k] = vdupq_n_f32(0);
                c1[k] = vdupq_n_f32(0);
                c2[k] = vdupq_n_f32(0);
                c3[k] = vdupq_n_f32(0);
            }

            for (size_t p = 0; p < K; p += 4) {
                float a0_0 = A[(i + 0) * K + p + 0], a1_0 = A[(i + 1) * K + p + 0], a2_0 = A[(i + 2) * K + p + 0], a3_0 = A[(i + 3) * K + p + 0];
                float a0_1 = A[(i + 0) * K + p + 1], a1_1 = A[(i + 1) * K + p + 1], a2_1 = A[(i + 2) * K + p + 1], a3_1 = A[(i + 3) * K + p + 1];
                float a0_2 = A[(i + 0) * K + p + 2], a1_2 = A[(i + 1) * K + p + 2], a2_2 = A[(i + 2) * K + p + 2], a3_2 = A[(i + 3) * K + p + 2];
                float a0_3 = A[(i + 0) * K + p + 3], a1_3 = A[(i + 1) * K + p + 3], a2_3 = A[(i + 2) * K + p + 3], a3_3 = A[(i + 3) * K + p + 3];

                float32x4_t va0 = {a0_0, a1_0, a2_0, a3_0};
                float32x4_t va1 = {a0_1, a1_1, a2_1, a3_1};
                float32x4_t va2 = {a0_2, a1_2, a2_2, a3_2};
                float32x4_t va3 = {a0_3, a1_3, a2_3, a3_3};

                uint32_t pk0 = B[(p + 0) * K_ints + j_int];
                uint32_t pk1 = B[(p + 1) * K_ints + j_int];
                uint32_t pk2 = B[(p + 2) * K_ints + j_int];
                uint32_t pk3 = B[(p + 3) * K_ints + j_int];

                for (int b = 0; b < 32; b += 4) {
                    uint32_t mask0 = 1u << (b + 0);
                    uint32_t mask1 = 1u << (b + 1);
                    uint32_t mask2 = 1u << (b + 2);
                    uint32_t mask3 = 1u << (b + 3);

                    // Not using NEON for bits extraction here as typical auto-vectorization worked very well.
                    // Doing a manual NEON vectorization per `p` could restrict flexibility compared to `c0[b]` arrays.
                }
            }
        }
    }
}
