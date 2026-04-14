#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

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
            float c0[32] = {0};
            float c1[32] = {0};
            float c2[32] = {0};
            float c3[32] = {0};

            for (size_t p = 0; p < K; p += 8) {
                float a0_0 = A[(i + 0) * K + p + 0], a1_0 = A[(i + 1) * K + p + 0], a2_0 = A[(i + 2) * K + p + 0], a3_0 = A[(i + 3) * K + p + 0];
                float a0_1 = A[(i + 0) * K + p + 1], a1_1 = A[(i + 1) * K + p + 1], a2_1 = A[(i + 2) * K + p + 1], a3_1 = A[(i + 3) * K + p + 1];
                float a0_2 = A[(i + 0) * K + p + 2], a1_2 = A[(i + 1) * K + p + 2], a2_2 = A[(i + 2) * K + p + 2], a3_2 = A[(i + 3) * K + p + 2];
                float a0_3 = A[(i + 0) * K + p + 3], a1_3 = A[(i + 1) * K + p + 3], a2_3 = A[(i + 2) * K + p + 3], a3_3 = A[(i + 3) * K + p + 3];
                float a0_4 = A[(i + 0) * K + p + 4], a1_4 = A[(i + 1) * K + p + 4], a2_4 = A[(i + 2) * K + p + 4], a3_4 = A[(i + 3) * K + p + 4];
                float a0_5 = A[(i + 0) * K + p + 5], a1_5 = A[(i + 1) * K + p + 5], a2_5 = A[(i + 2) * K + p + 5], a3_5 = A[(i + 3) * K + p + 5];
                float a0_6 = A[(i + 0) * K + p + 6], a1_6 = A[(i + 1) * K + p + 6], a2_6 = A[(i + 2) * K + p + 6], a3_6 = A[(i + 3) * K + p + 6];
                float a0_7 = A[(i + 0) * K + p + 7], a1_7 = A[(i + 1) * K + p + 7], a2_7 = A[(i + 2) * K + p + 7], a3_7 = A[(i + 3) * K + p + 7];

                uint32_t pk0 = B[(p + 0) * K_ints + j_int];
                uint32_t pk1 = B[(p + 1) * K_ints + j_int];
                uint32_t pk2 = B[(p + 2) * K_ints + j_int];
                uint32_t pk3 = B[(p + 3) * K_ints + j_int];
                uint32_t pk4 = B[(p + 4) * K_ints + j_int];
                uint32_t pk5 = B[(p + 5) * K_ints + j_int];
                uint32_t pk6 = B[(p + 6) * K_ints + j_int];
                uint32_t pk7 = B[(p + 7) * K_ints + j_int];

                #pragma GCC unroll 32
                for (int b = 0; b < 32; ++b) {
                    uint32_t mask = 1u << b;
                    
                    float add0_0 = (pk0 & mask) ? a0_0 : 0.0f;
                    float add1_0 = (pk0 & mask) ? a1_0 : 0.0f;
                    float add2_0 = (pk0 & mask) ? a2_0 : 0.0f;
                    float add3_0 = (pk0 & mask) ? a3_0 : 0.0f;

                    float add0_1 = (pk1 & mask) ? a0_1 : 0.0f;
                    float add1_1 = (pk1 & mask) ? a1_1 : 0.0f;
                    float add2_1 = (pk1 & mask) ? a2_1 : 0.0f;
                    float add3_1 = (pk1 & mask) ? a3_1 : 0.0f;

                    float add0_2 = (pk2 & mask) ? a0_2 : 0.0f;
                    float add1_2 = (pk2 & mask) ? a1_2 : 0.0f;
                    float add2_2 = (pk2 & mask) ? a2_2 : 0.0f;
                    float add3_2 = (pk2 & mask) ? a3_2 : 0.0f;

                    float add0_3 = (pk3 & mask) ? a0_3 : 0.0f;
                    float add1_3 = (pk3 & mask) ? a1_3 : 0.0f;
                    float add2_3 = (pk3 & mask) ? a2_3 : 0.0f;
                    float add3_3 = (pk3 & mask) ? a3_3 : 0.0f;

                    float add0_4 = (pk4 & mask) ? a0_4 : 0.0f;
                    float add1_4 = (pk4 & mask) ? a1_4 : 0.0f;
                    float add2_4 = (pk4 & mask) ? a2_4 : 0.0f;
                    float add3_4 = (pk4 & mask) ? a3_4 : 0.0f;

                    float add0_5 = (pk5 & mask) ? a0_5 : 0.0f;
                    float add1_5 = (pk5 & mask) ? a1_5 : 0.0f;
                    float add2_5 = (pk5 & mask) ? a2_5 : 0.0f;
                    float add3_5 = (pk5 & mask) ? a3_5 : 0.0f;

                    float add0_6 = (pk6 & mask) ? a0_6 : 0.0f;
                    float add1_6 = (pk6 & mask) ? a1_6 : 0.0f;
                    float add2_6 = (pk6 & mask) ? a2_6 : 0.0f;
                    float add3_6 = (pk6 & mask) ? a3_6 : 0.0f;

                    float add0_7 = (pk7 & mask) ? a0_7 : 0.0f;
                    float add1_7 = (pk7 & mask) ? a1_7 : 0.0f;
                    float add2_7 = (pk7 & mask) ? a2_7 : 0.0f;
                    float add3_7 = (pk7 & mask) ? a3_7 : 0.0f;

                    c0[b] += add0_0 + add0_1 + add0_2 + add0_3 + add0_4 + add0_5 + add0_6 + add0_7;
                    c1[b] += add1_0 + add1_1 + add1_2 + add1_3 + add1_4 + add1_5 + add1_6 + add1_7;
                    c2[b] += add2_0 + add2_1 + add2_2 + add2_3 + add2_4 + add2_5 + add2_6 + add2_7;
                    c3[b] += add3_0 + add3_1 + add3_2 + add3_3 + add3_4 + add3_5 + add3_6 + add3_7;
                }
            }
            
            for (int b = 0; b < 32; ++b) {
                C[(i + 0) * K + j_int * 32 + b] = 2.0f * c0[b] - row_sums[i + 0];
                C[(i + 1) * K + j_int * 32 + b] = 2.0f * c1[b] - row_sums[i + 1];
                C[(i + 2) * K + j_int * 32 + b] = 2.0f * c2[b] - row_sums[i + 2];
                C[(i + 3) * K + j_int * 32 + b] = 2.0f * c3[b] - row_sums[i + 3];
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
                    uint32_t mask = 1u << b;
                    c[b] += (packed & mask) ? a : 0.0f;
                }
            }
            for (int b = 0; b < 32; ++b) {
                C[i * K + j_int * 32 + b] = 2.0f * c[b] - row_sums[i];
            }
        }
    }
}
