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
    for (; i + 7 < M; i += 8) {
        for (size_t j_int = 0; j_int < K_ints; ++j_int) {
            
            for (int b_vec = 0; b_vec < 32; b_vec += 4) {
                float c0[4] = {0}; float c1[4] = {0}; float c2[4] = {0}; float c3[4] = {0};
                float c4[4] = {0}; float c5[4] = {0}; float c6[4] = {0}; float c7[4] = {0};

                for (size_t p = 0; p < K; ++p) {
                    float a0 = A[(i + 0) * K + p];
                    float a1 = A[(i + 1) * K + p];
                    float a2 = A[(i + 2) * K + p];
                    float a3 = A[(i + 3) * K + p];
                    float a4 = A[(i + 4) * K + p];
                    float a5 = A[(i + 5) * K + p];
                    float a6 = A[(i + 6) * K + p];
                    float a7 = A[(i + 7) * K + p];

                    uint32_t pk = B[p * K_ints + j_int];
                    
                    float fb0 = (pk & (1u << (b_vec + 0))) ? 1.0f : 0.0f;
                    float fb1 = (pk & (1u << (b_vec + 1))) ? 1.0f : 0.0f;
                    float fb2 = (pk & (1u << (b_vec + 2))) ? 1.0f : 0.0f;
                    float fb3 = (pk & (1u << (b_vec + 3))) ? 1.0f : 0.0f;

                    // manually unroll b inside
                    c0[0] += fb0 * a0; c0[1] += fb1 * a0; c0[2] += fb2 * a0; c0[3] += fb3 * a0;
                    c1[0] += fb0 * a1; c1[1] += fb1 * a1; c1[2] += fb2 * a1; c1[3] += fb3 * a1;
                    c2[0] += fb0 * a2; c2[1] += fb1 * a2; c2[2] += fb2 * a2; c2[3] += fb3 * a2;
                    c3[0] += fb0 * a3; c3[1] += fb1 * a3; c3[2] += fb2 * a3; c3[3] += fb3 * a3;
                    
                    c4[0] += fb0 * a4; c4[1] += fb1 * a4; c4[2] += fb2 * a4; c4[3] += fb3 * a4;
                    c5[0] += fb0 * a5; c5[1] += fb1 * a5; c5[2] += fb2 * a5; c5[3] += fb3 * a5;
                    c6[0] += fb0 * a6; c6[1] += fb1 * a6; c6[2] += fb2 * a6; c6[3] += fb3 * a6;
                    c7[0] += fb0 * a7; c7[1] += fb1 * a7; c7[2] += fb2 * a7; c7[3] += fb3 * a7;
                }
                
                for (int b = 0; b < 4; ++b) {
                    C[(i + 0) * K + j_int * 32 + b_vec + b] = 2.0f * c0[b] - row_sums[i + 0];
                    C[(i + 1) * K + j_int * 32 + b_vec + b] = 2.0f * c1[b] - row_sums[i + 1];
                    C[(i + 2) * K + j_int * 32 + b_vec + b] = 2.0f * c2[b] - row_sums[i + 2];
                    C[(i + 3) * K + j_int * 32 + b_vec + b] = 2.0f * c3[b] - row_sums[i + 3];
                    C[(i + 4) * K + j_int * 32 + b_vec + b] = 2.0f * c4[b] - row_sums[i + 4];
                    C[(i + 5) * K + j_int * 32 + b_vec + b] = 2.0f * c5[b] - row_sums[i + 5];
                    C[(i + 6) * K + j_int * 32 + b_vec + b] = 2.0f * c6[b] - row_sums[i + 6];
                    C[(i + 7) * K + j_int * 32 + b_vec + b] = 2.0f * c7[b] - row_sums[i + 7];
                }
            }
        }
    }
    
    // Handle remaining M rows
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
