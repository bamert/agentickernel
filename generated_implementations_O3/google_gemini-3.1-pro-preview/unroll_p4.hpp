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
            
            size_t p = 0;
            for (; p + 3 < K; p += 4) {
                float a0 = A[i * K + p];
                float a1 = A[i * K + p + 1];
                float a2 = A[i * K + p + 2];
                float a3 = A[i * K + p + 3];

                uint32_t p0 = B[(p + 0) * K_ints + j_word];
                uint32_t p1 = B[(p + 1) * K_ints + j_word];
                uint32_t p2 = B[(p + 2) * K_ints + j_word];
                uint32_t p3 = B[(p + 3) * K_ints + j_word];
                
                #pragma unroll
                for (int b = 0; b < 32; ++b) {
                    uint32_t mask = 1u << b;
                    float s = sums[b];
                    if (p0 & mask) s += a0;
                    if (p1 & mask) s += a1;
                    if (p2 & mask) s += a2;
                    if (p3 & mask) s += a3;
                    sums[b] = s;
                }
            }
            
            for (; p < K; ++p) {
                float a_val = A[i * K + p];
                uint32_t packed = B[p * K_ints + j_word];
                
                #pragma unroll
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
