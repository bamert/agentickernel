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
            
            for (int b_blk = 0; b_blk < 32; b_blk += 16) {
                float c0[16] = {0};
                float c1[16] = {0};
                float c2[16] = {0};
                float c3[16] = {0};

                for (size_t p = 0; p < K; p += 4) {
                    uint32_t pk0 = B[(p + 0) * K_ints + j_int];
                    uint32_t pk1 = B[(p + 1) * K_ints + j_int];
                    uint32_t pk2 = B[(p + 2) * K_ints + j_int];
                    uint32_t pk3 = B[(p + 3) * K_ints + j_int];

                    float a0_0 = A[(i + 0) * K + p + 0];
                    float a1_0 = A[(i + 1) * K + p + 0];
                    float a2_0 = A[(i + 2) * K + p + 0];
                    float a3_0 = A[(i + 3) * K + p + 0];

                    float a0_1 = A[(i + 0) * K + p + 1];
                    float a1_1 = A[(i + 1) * K + p + 1];
                    float a2_1 = A[(i + 2) * K + p + 1];
                    float a3_1 = A[(i + 3) * K + p + 1];

                    float a0_2 = A[(i + 0) * K + p + 2];
                    float a1_2 = A[(i + 1) * K + p + 2];
                    float a2_2 = A[(i + 2) * K + p + 2];
                    float a3_2 = A[(i + 3) * K + p + 2];

                    float a0_3 = A[(i + 0) * K + p + 3];
                    float a1_3 = A[(i + 1) * K + p + 3];
                    float a2_3 = A[(i + 2) * K + p + 3];
                    float a3_3 = A[(i + 3) * K + p + 3];

                    #pragma GCC unroll 16
                    for (int b = 0; b < 16; ++b) {
                        int shift = b_blk + b;
                        float fb0 = (pk0 >> shift) & 1;
                        float fb1 = (pk1 >> shift) & 1;
                        float fb2 = (pk2 >> shift) & 1;
                        float fb3 = (pk3 >> shift) & 1;

                        c0[b] += fb0 * a0_0 + fb1 * a0_1 + fb2 * a0_2 + fb3 * a0_3;
                        c1[b] += fb0 * a1_0 + fb1 * a1_1 + fb2 * a1_2 + fb3 * a1_3;
                        c2[b] += fb0 * a2_0 + fb1 * a2_1 + fb2 * a2_2 + fb3 * a2_3;
                        c3[b] += fb0 * a3_0 + fb1 * a3_1 + fb2 * a3_2 + fb3 * a3_3;
                    }
                }
                
                for (int b = 0; b < 16; ++b) {
                    C[(i + 0) * K + j_int * 32 + b_blk + b] = 2.0f * c0[b] - row_sums[i + 0];
                    C[(i + 1) * K + j_int * 32 + b_blk + b] = 2.0f * c1[b] - row_sums[i + 1];
                    C[(i + 2) * K + j_int * 32 + b_blk + b] = 2.0f * c2[b] - row_sums[i + 2];
                    C[(i + 3) * K + j_int * 32 + b_blk + b] = 2.0f * c3[b] - row_sums[i + 3];
                }
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
