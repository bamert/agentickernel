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
    // Unrolling M by 6 gives theoretically more register utilization.
    for (; i + 5 < M; i += 6) {
        for (size_t j_int = 0; j_int < K_ints; ++j_int) {
            float c0[32] = {0}; float c1[32] = {0}; float c2[32] = {0};
            float c3[32] = {0}; float c4[32] = {0}; float c5[32] = {0};

            for (size_t p = 0; p < K; p += 4) {
                float a0_0 = A[(i + 0) * K + p + 0], a1_0 = A[(i + 1) * K + p + 0], a2_0 = A[(i + 2) * K + p + 0], a3_0 = A[(i + 3) * K + p + 0], a4_0 = A[(i + 4) * K + p + 0], a5_0 = A[(i + 5) * K + p + 0];
                float a0_1 = A[(i + 0) * K + p + 1], a1_1 = A[(i + 1) * K + p + 1], a2_1 = A[(i + 2) * K + p + 1], a3_1 = A[(i + 3) * K + p + 1], a4_1 = A[(i + 4) * K + p + 1], a5_1 = A[(i + 5) * K + p + 1];
                float a0_2 = A[(i + 0) * K + p + 2], a1_2 = A[(i + 1) * K + p + 2], a2_2 = A[(i + 2) * K + p + 2], a3_2 = A[(i + 3) * K + p + 2], a4_2 = A[(i + 4) * K + p + 2], a5_2 = A[(i + 5) * K + p + 2];
                float a0_3 = A[(i + 0) * K + p + 3], a1_3 = A[(i + 1) * K + p + 3], a2_3 = A[(i + 2) * K + p + 3], a3_3 = A[(i + 3) * K + p + 3], a4_3 = A[(i + 4) * K + p + 3], a5_3 = A[(i + 5) * K + p + 3];

                uint32_t pk0 = B[(p + 0) * K_ints + j_int];
                uint32_t pk1 = B[(p + 1) * K_ints + j_int];
                uint32_t pk2 = B[(p + 2) * K_ints + j_int];
                uint32_t pk3 = B[(p + 3) * K_ints + j_int];

                #pragma GCC unroll 32
                for (int b = 0; b < 32; ++b) {
                    float fb0 = (pk0 & 1); pk0 >>= 1;
                    float fb1 = (pk1 & 1); pk1 >>= 1;
                    float fb2 = (pk2 & 1); pk2 >>= 1;
                    float fb3 = (pk3 & 1); pk3 >>= 1;

                    c0[b] += fb0 * a0_0 + fb1 * a0_1 + fb2 * a0_2 + fb3 * a0_3;
                    c1[b] += fb0 * a1_0 + fb1 * a1_1 + fb2 * a1_2 + fb3 * a1_3;
                    c2[b] += fb0 * a2_0 + fb1 * a2_1 + fb2 * a2_2 + fb3 * a2_3;
                    c3[b] += fb0 * a3_0 + fb1 * a3_1 + fb2 * a3_2 + fb3 * a3_3;
                    c4[b] += fb0 * a4_0 + fb1 * a4_1 + fb2 * a4_2 + fb3 * a4_3;
                    c5[b] += fb0 * a5_0 + fb1 * a5_1 + fb2 * a5_2 + fb3 * a5_3;
                }
            }
            
            float rs0 = row_sums[i + 0]; float rs1 = row_sums[i + 1]; float rs2 = row_sums[i + 2];
            float rs3 = row_sums[i + 3]; float rs4 = row_sums[i + 4]; float rs5 = row_sums[i + 5];

            float* C_ptr0 = &C[(i + 0) * K + j_int * 32]; float* C_ptr1 = &C[(i + 1) * K + j_int * 32];
            float* C_ptr2 = &C[(i + 2) * K + j_int * 32]; float* C_ptr3 = &C[(i + 3) * K + j_int * 32];
            float* C_ptr4 = &C[(i + 4) * K + j_int * 32]; float* C_ptr5 = &C[(i + 5) * K + j_int * 32];

            for (int b = 0; b < 32; ++b) {
                C_ptr0[b] = 2.0f * c0[b] - rs0;
                C_ptr1[b] = 2.0f * c1[b] - rs1;
                C_ptr2[b] = 2.0f * c2[b] - rs2;
                C_ptr3[b] = 2.0f * c3[b] - rs3;
                C_ptr4[b] = 2.0f * c4[b] - rs4;
                C_ptr5[b] = 2.0f * c5[b] - rs5;
            }
        }
    }
    
    // Fall back to M unroll 4
    for (; i + 3 < M; i += 4) {
        for (size_t j_int = 0; j_int < K_ints; ++j_int) {
            float c0[32] = {0}; float c1[32] = {0}; float c2[32] = {0}; float c3[32] = {0};

            for (size_t p = 0; p < K; p += 4) {
                float a0_0 = A[(i + 0) * K + p + 0], a1_0 = A[(i + 1) * K + p + 0], a2_0 = A[(i + 2) * K + p + 0], a3_0 = A[(i + 3) * K + p + 0];
                float a0_1 = A[(i + 0) * K + p + 1], a1_1 = A[(i + 1) * K + p + 1], a2_1 = A[(i + 2) * K + p + 1], a3_1 = A[(i + 3) * K + p + 1];
                float a0_2 = A[(i + 0) * K + p + 2], a1_2 = A[(i + 1) * K + p + 2], a2_2 = A[(i + 2) * K + p + 2], a3_2 = A[(i + 3) * K + p + 2];
                float a0_3 = A[(i + 0) * K + p + 3], a1_3 = A[(i + 1) * K + p + 3], a2_3 = A[(i + 2) * K + p + 3], a3_3 = A[(i + 3) * K + p + 3];

                uint32_t pk0 = B[(p + 0) * K_ints + j_int], pk1 = B[(p + 1) * K_ints + j_int];
                uint32_t pk2 = B[(p + 2) * K_ints + j_int], pk3 = B[(p + 3) * K_ints + j_int];

                #pragma GCC unroll 32
                for (int b = 0; b < 32; ++b) {
                    float fb0 = (pk0 & 1); pk0 >>= 1;
                    float fb1 = (pk1 & 1); pk1 >>= 1;
                    float fb2 = (pk2 & 1); pk2 >>= 1;
                    float fb3 = (pk3 & 1); pk3 >>= 1;

                    c0[b] += fb0 * a0_0 + fb1 * a0_1 + fb2 * a0_2 + fb3 * a0_3;
                    c1[b] += fb0 * a1_0 + fb1 * a1_1 + fb2 * a1_2 + fb3 * a1_3;
                    c2[b] += fb0 * a2_0 + fb1 * a2_1 + fb2 * a2_2 + fb3 * a2_3;
                    c3[b] += fb0 * a3_0 + fb1 * a3_1 + fb2 * a3_2 + fb3 * a3_3;
                }
            }
            float rs0 = row_sums[i + 0], rs1 = row_sums[i + 1], rs2 = row_sums[i + 2], rs3 = row_sums[i + 3];
            float* C_ptr0 = &C[(i + 0) * K + j_int * 32]; float* C_ptr1 = &C[(i + 1) * K + j_int * 32];
            float* C_ptr2 = &C[(i + 2) * K + j_int * 32]; float* C_ptr3 = &C[(i + 3) * K + j_int * 32];

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
                    float fbit = (packed & 1); packed >>= 1;
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
