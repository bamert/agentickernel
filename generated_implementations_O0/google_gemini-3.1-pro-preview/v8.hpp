#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    std::vector<float> row_sums(M, 0.0f);
    for (size_t i = 0; i < M; ++i) {
        float sum = 0;
        for (size_t p = 0; p < K; ++p) {
            sum += A[i * K + p];
        }
        row_sums[i] = sum;
    }

    size_t i = 0;
    // Unroll M by 2 to prevent register spilling (64 floats = 16 vec registers, 16 remaining for compute)
    for (; i + 1 < M; i += 2) {
        for (size_t j_int = 0; j_int < K_ints; ++j_int) {
            float c0[32] = {0};
            float c1[32] = {0};

            for (size_t p = 0; p < K; ++p) {
                float a0 = A[(i + 0) * K + p];
                float a1 = A[(i + 1) * K + p];
                
                uint32_t packed = B[p * K_ints + j_int];
                
                #pragma GCC unroll 32
                for (int b = 0; b < 32; ++b) {
                    float fbit = (packed >> b) & 1;
                    c0[b] += fbit * a0;
                    c1[b] += fbit * a1;
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
                #pragma GCC unroll 32
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
