#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }
    }

    for (size_t i = 0; i < M; ++i) {
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            float a_neg = -a_val;
            
            for (size_t j_blk = 0; j_blk < K_ints; ++j_blk) {
                uint32_t packed = B[p * K_ints + j_blk];
                size_t c_idx = i * K + j_blk * 32;
                
                for (int b = 0; b < 32; ++b) {
                    float sign_val = ((packed >> b) & 1) ? a_val : a_neg;
                    C[c_idx + b] += sign_val;
                }
            }
        }
    }
}
