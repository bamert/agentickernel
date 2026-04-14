#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; i++) {
        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) row_sum += A[i * K + p];
        for (size_t j = 0; j < K; ++j) C[i * K + j] = -row_sum;
    }

    size_t i = 0;
    for (; i + 3 < M; i += 4) {
        for (size_t j_word = 0; j_word < K_ints; ++j_word) {
            float sums0[32] = {0};
            float sums1[32] = {0};
            float sums2[32] = {0};
            float sums3[32] = {0};
            
            for (size_t p = 0; p < K; ++p) {
                float a0 = A[(i+0) * K + p];
                float a1 = A[(i+1) * K + p];
                float a2 = A[(i+2) * K + p];
                float a3 = A[(i+3) * K + p];

                uint32_t packed = B[p * K_ints + j_word];
                
                for (int b = 0; b < 32; ++b) {
                    if ((packed >> b) & 1) {
                        sums0[b] += a0;
                        sums1[b] += a1;
                        sums2[b] += a2;
                        sums3[b] += a3;
                    }
                }
            }
            
            for (int b = 0; b < 32; ++b) {
                C[(i+0) * K + j_word * 32 + b] += sums0[b] * 2.0f;
                C[(i+1) * K + j_word * 32 + b] += sums1[b] * 2.0f;
                C[(i+2) * K + j_word * 32 + b] += sums2[b] * 2.0f;
                C[(i+3) * K + j_word * 32 + b] += sums3[b] * 2.0f;
            }
        }
    }

    for (; i < M; ++i) {
        for (size_t j_word = 0; j_word < K_ints; ++j_word) {
            float sums0[32] = {0};
            for (size_t p = 0; p < K; ++p) {
                float a0 = A[i * K + p];
                uint32_t packed = B[p * K_ints + j_word];
                for (int b = 0; b < 32; ++b) {
                    if ((packed >> b) & 1) {
                        sums0[b] += a0;
                    }
                }
            }
            for (int b = 0; b < 32; ++b) {
                C[i * K + j_word * 32 + b] += sums0[b] * 2.0f;
            }
        }
    }
}
