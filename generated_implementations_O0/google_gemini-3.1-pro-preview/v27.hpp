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
                    float fb0 = (pk0 & mask) ? 1.0f : 0.0f;
                    float fb1 = (pk1 & mask) ? 1.0f : 0.0f;
                    float fb2 = (pk2 & mask) ? 1.0f : 0.0f;
                    float fb3 = (pk3 & mask) ? 1.0f : 0.0f;
                    float fb4 = (pk4 & mask) ? 1.0f : 0.0f;
                    float fb5 = (pk5 & mask) ? 1.0f : 0.0f;
                    float fb6 = (pk6 & mask) ? 1.0f : 0.0f;
                    float fb7 = (pk7 & mask) ? 1.0f : 0.0f;

                    c0[b] += fb0 * a0_0 + fb1 * a0_1 + fb2 * a0_2 + fb3 * a0_3 + fb4 * a0_4 + fb5 * a0_5 + fb6 * a0_6 + fb7 * a0_7;
                    c1[b] += fb0 * a1_0 + fb1 * a1_1 + fb2 * a1_2 + fb3 * a1_3 + fb4 * a1_4 + fb5 * a1_5 + fb6 * a1_6 + fb7 * a1_7;
                    c2[b] += fb0 * a2_0 + fb1 * a2_1 + fb2 * a2_2 + fb3 * a2_3 + fb4 * a2_4 + fb5 * a2_5 + fb6 * a2_6 + fb7 * a2_7;
                    c3[b] += fb0 * a3_0 + fb1 * a3_1 + fb2 * a3_2 + fb3 * a3_3 + fb4 * a3_4 + fb5 * a3_5 + fb6 * a3_6 + fb7 * a3_7;
                }
            }
            
            float* C_ptr0 = &C[(i + 0) * K + j_int * 32];
            float* C_ptr1 = &C[(i + 1) * K + j_int * 32];
            float* C_ptr2 = &C[(i + 2) * K + j_int * 32];
            float* C_ptr3 = &C[(i + 3) * K + j_int * 32];
            
            float rs0 = row_sums[i + 0];
            float rs1 = row_sums[i + 1];
            float rs2 = row_sums[i + 2];
            float rs3 = row_sums[i + 3];

            for (int b = 0; b < 32; ++b) {
                C_ptr0[b] = 2.0f * c0[b] - rs0;
                C_ptr1[b] = 2.0f * c1[b] - rs1;
                C_ptr2[b] = 2.0f * c2[b] - rs2;
                C_ptr3[b] = 2.0f * c3[b] - rs3;
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
                    float fbit = (packed >> b) & 1;
                    c[b] += fbit * a;
                }
            }
            for (int b = 0; b < 32; ++b) {
                C[i * K + j_int * 32 + b] = 2.0f * c[b] - row_sums[i];
            }
        }
    }
}
