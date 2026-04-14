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
                    float fb0 = (float)(pk0 & 1); pk0 >>= 1;
                    float fb1 = (float)(pk1 & 1); pk1 >>= 1;
                    float fb2 = (float)(pk2 & 1); pk2 >>= 1;
                    float fb3 = (float)(pk3 & 1); pk3 >>= 1;
                    float fb4 = (float)(pk4 & 1); pk4 >>= 1;
                    float fb5 = (float)(pk5 & 1); pk5 >>= 1;
                    float fb6 = (float)(pk6 & 1); pk6 >>= 1;
                    float fb7 = (float)(pk7 & 1); pk7 >>= 1;

                    c0[b] += fb0 * a0_0 + fb1 * a0_1 + fb2 * a0_2 + fb3 * a0_3 + fb4 * a0_4 + fb5 * a0_5 + fb6 * a0_6 + fb7 * a0_7;
                    c1[b] += fb0 * a1_0 + fb1 * a1_1 + fb2 * a1_2 + fb3 * a1_3 + fb4 * a1_4 + fb5 * a1_5 + fb6 * a1_6 + fb7 * a1_7;
                    c2[b] += fb0 * a2_0 + fb1 * a2_1 + fb2 * a2_2 + fb3 * a2_3 + fb4 * a2_4 + fb5 * a2_5 + fb6 * a2_6 + fb7 * a2_7;
                    c3[b] += fb0 * a3_0 + fb1 * a3_1 + fb2 * a3_2 + fb3 * a3_3 + fb4 * a3_4 + fb5 * a3_5 + fb6 * a3_6 + fb7 * a3_7;
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
                    float fbit = (float)(packed & 1);
                    packed >>= 1;
                    c[b] += fbit * a;
                }
            }
            for (int b = 0; b < 32; ++b) {
                C[i * K + j_int * 32 + b] = 2.0f * c[b] - row_sums[i];
            }
        }
    }
}
