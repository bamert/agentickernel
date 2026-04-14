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
    for (; i + 1 < M; i += 2) {
        for (size_t j_int = 0; j_int < K_ints; ++j_int) {
            float c0[32] = {0};
            float c1[32] = {0};

            for (size_t p = 0; p < K; p += 16) {
                float a0_0 = A[(i+0)*K+p+0], a1_0 = A[(i+1)*K+p+0];
                float a0_1 = A[(i+0)*K+p+1], a1_1 = A[(i+1)*K+p+1];
                float a0_2 = A[(i+0)*K+p+2], a1_2 = A[(i+1)*K+p+2];
                float a0_3 = A[(i+0)*K+p+3], a1_3 = A[(i+1)*K+p+3];
                float a0_4 = A[(i+0)*K+p+4], a1_4 = A[(i+1)*K+p+4];
                float a0_5 = A[(i+0)*K+p+5], a1_5 = A[(i+1)*K+p+5];
                float a0_6 = A[(i+0)*K+p+6], a1_6 = A[(i+1)*K+p+6];
                float a0_7 = A[(i+0)*K+p+7], a1_7 = A[(i+1)*K+p+7];
                float a0_8 = A[(i+0)*K+p+8], a1_8 = A[(i+1)*K+p+8];
                float a0_9 = A[(i+0)*K+p+9], a1_9 = A[(i+1)*K+p+9];
                float a0_10 = A[(i+0)*K+p+10], a1_10 = A[(i+1)*K+p+10];
                float a0_11 = A[(i+0)*K+p+11], a1_11 = A[(i+1)*K+p+11];
                float a0_12 = A[(i+0)*K+p+12], a1_12 = A[(i+1)*K+p+12];
                float a0_13 = A[(i+0)*K+p+13], a1_13 = A[(i+1)*K+p+13];
                float a0_14 = A[(i+0)*K+p+14], a1_14 = A[(i+1)*K+p+14];
                float a0_15 = A[(i+0)*K+p+15], a1_15 = A[(i+1)*K+p+15];

                uint32_t pk0 = B[(p+0)*K_ints+j_int];
                uint32_t pk1 = B[(p+1)*K_ints+j_int];
                uint32_t pk2 = B[(p+2)*K_ints+j_int];
                uint32_t pk3 = B[(p+3)*K_ints+j_int];
                uint32_t pk4 = B[(p+4)*K_ints+j_int];
                uint32_t pk5 = B[(p+5)*K_ints+j_int];
                uint32_t pk6 = B[(p+6)*K_ints+j_int];
                uint32_t pk7 = B[(p+7)*K_ints+j_int];
                uint32_t pk8 = B[(p+8)*K_ints+j_int];
                uint32_t pk9 = B[(p+9)*K_ints+j_int];
                uint32_t pk10 = B[(p+10)*K_ints+j_int];
                uint32_t pk11 = B[(p+11)*K_ints+j_int];
                uint32_t pk12 = B[(p+12)*K_ints+j_int];
                uint32_t pk13 = B[(p+13)*K_ints+j_int];
                uint32_t pk14 = B[(p+14)*K_ints+j_int];
                uint32_t pk15 = B[(p+15)*K_ints+j_int];

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
                    float fb8 = (pk8 & mask) ? 1.0f : 0.0f;
                    float fb9 = (pk9 & mask) ? 1.0f : 0.0f;
                    float fb10 = (pk10 & mask) ? 1.0f : 0.0f;
                    float fb11 = (pk11 & mask) ? 1.0f : 0.0f;
                    float fb12 = (pk12 & mask) ? 1.0f : 0.0f;
                    float fb13 = (pk13 & mask) ? 1.0f : 0.0f;
                    float fb14 = (pk14 & mask) ? 1.0f : 0.0f;
                    float fb15 = (pk15 & mask) ? 1.0f : 0.0f;

                    c0[b] += fb0 * a0_0 + fb1 * a0_1 + fb2 * a0_2 + fb3 * a0_3 + fb4 * a0_4 + fb5 * a0_5 + fb6 * a0_6 + fb7 * a0_7 +
                             fb8 * a0_8 + fb9 * a0_9 + fb10 * a0_10 + fb11 * a0_11 + fb12 * a0_12 + fb13 * a0_13 + fb14 * a0_14 + fb15 * a0_15;
                    
                    c1[b] += fb0 * a1_0 + fb1 * a1_1 + fb2 * a1_2 + fb3 * a1_3 + fb4 * a1_4 + fb5 * a1_5 + fb6 * a1_6 + fb7 * a1_7 +
                             fb8 * a1_8 + fb9 * a1_9 + fb10 * a1_10 + fb11 * a1_11 + fb12 * a1_12 + fb13 * a1_13 + fb14 * a1_14 + fb15 * a1_15;
                }
            }
            
            for (int b = 0; b < 32; ++b) {
                C[(i + 0) * K + j_int * 32 + b] = 2.0f * c0[b] - row_sums[i + 0];
                C[(i + 1) * K + j_int * 32 + b] = 2.0f * c1[b] - row_sums[i + 1];
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
