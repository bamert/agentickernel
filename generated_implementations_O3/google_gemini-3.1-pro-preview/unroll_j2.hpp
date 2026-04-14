#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        size_t j_word = 0;
        for (; j_word + 1 < K_ints; j_word += 2) {
            float sums0[32] = {0.0f};
            float sums1[32] = {0.0f};
            
            for (size_t p = 0; p < K; ++p) {
                float a_val = A[i * K + p];
                uint32_t packed0 = B[p * K_ints + j_word];
                uint32_t packed1 = B[p * K_ints + j_word + 1];
                
                #pragma unroll
                for (int b = 0; b < 32; ++b) {
                    float sign0 = ((packed0 >> b) & 1) ? 1.0f : -1.0f;
                    float sign1 = ((packed1 >> b) & 1) ? 1.0f : -1.0f;
                    sums0[b] += a_val * sign0;
                    sums1[b] += a_val * sign1;
                }
            }
            
            for (int b = 0; b < 32; ++b) {
                C[i * K + j_word * 32 + b] = sums0[b];
                C[i * K + (j_word + 1) * 32 + b] = sums1[b];
            }
        }
        for (; j_word < K_ints; ++j_word) {
            float sums[32] = {0.0f};
            
            for (size_t p = 0; p < K; ++p) {
                float a_val = A[i * K + p];
                uint32_t packed = B[p * K_ints + j_word];
                
                #pragma unroll
                for (int b = 0; b < 32; ++b) {
                    float sign = ((packed >> b) & 1) ? 1.0f : -1.0f;
                    sums[b] += a_val * sign;
                }
            }
            
            for (int b = 0; b < 32; ++b) {
                C[i * K + j_word * 32 + b] = sums[b];
            }
        }
    }
}
