#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    size_t i = 0;
    for (; i + 3 < M; i += 4) {
        for (size_t j_int = 0; j_int < K_ints; ++j_int) {
            float c0[32] = {0};
            float c1[32] = {0};
            float c2[32] = {0};
            float c3[32] = {0};

            for (size_t p = 0; p < K; ++p) {
                float a0 = A[(i + 0) * K + p];
                float a1 = A[(i + 1) * K + p];
                float a2 = A[(i + 2) * K + p];
                float a3 = A[(i + 3) * K + p];
                
                uint32_t packed = B[p * K_ints + j_int];
                
                for (int b = 0; b < 32; ++b) {
                    uint32_t bit = (packed >> b) & 1;
                    float sign = bit ? 1.0f : -1.0f;
                    c0[b] += a0 * sign;
                    c1[b] += a1 * sign;
                    c2[b] += a2 * sign;
                    c3[b] += a3 * sign;
                }
            }
            
            for (int b = 0; b < 32; ++b) {
                C[(i + 0) * K + j_int * 32 + b] = c0[b];
                C[(i + 1) * K + j_int * 32 + b] = c1[b];
                C[(i + 2) * K + j_int * 32 + b] = c2[b];
                C[(i + 3) * K + j_int * 32 + b] = c3[b];
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
                    uint32_t bit = (packed >> b) & 1;
                    float sign = bit ? 1.0f : -1.0f;
                    c[b] += a * sign;
                }
            }
            
            for (int b = 0; b < 32; ++b) {
                C[i * K + j_int * 32 + b] = c[b];
            }
        }
    }
}
