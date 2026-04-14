#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M * K; ++i) {
        C[i] = 0.0f;
    }

    for (size_t i = 0; i < M; ++i) {
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const uint32_t* B_row = &B[p * K_ints];
            float* C_row = &C[i * K];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = B_row[j_int];
                
                for (int b = 0; b < 32; ++b) {
                    uint32_t bit = (packed >> b) & 1;
                    float sign = bit ? 1.0f : -1.0f;
                    C_row[j_int * 32 + b] += a_val * sign;
                }
            }
        }
    }
}
