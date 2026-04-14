#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Precompute signs for all B elements: convert B's packed bits to sign array
    float* signs = new float[K * K];
    for (size_t p = 0; p < K; ++p) {
        for (size_t j = 0; j < K; ++j) {
            uint32_t packed = B[p * K_ints + (j / 32)];
            uint32_t bit = (packed >> (j % 32)) & 1;
            signs[p * K + j] = bit ? 1.0f : -1.0f;
        }
    }

    // Now perform standard matrix multiplication with precomputed signs
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < K; ++p) {
                sum += A[i * K + p] * signs[p * K + j];
            }
            C[i * K + j] = sum;
        }
    }

    delete[] signs;
}
