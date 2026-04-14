#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            row_sum += A[i * K + p];
        }

        for (size_t j_word = 0; j_word < K_ints; ++j_word) {
            float sums[32] = {0.0f};
            
            for (size_t p = 0; p < K; ++p) {
                float a_val = A[i * K + p];
                uint32_t packed = B[p * K_ints + j_word];
                
                for (int b = 0; b < 32; ++b) {
                    if ((packed >> b) & 1) {
                        sums[b] += a_val;
                    }
                }
            }
            
            for (int b = 0; b < 32; ++b) {
                C[i * K + j_word * 32 + b] = sums[b] * 2.0f - row_sum;
            }
        }
    }
}
