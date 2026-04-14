#pragma once
#include <cstdint>
#include <cstddef>

// Key insight: Instead of extracting individual bits for each (i,j,p), 
// we can iterate over packed words and process 32 bits at a time.
// For each row i of A and each row p of B, we process 32 columns of B at once.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Zero out C
    for (size_t i = 0; i < M * K; ++i) {
        C[i] = 0.0f;
    }

    for (size_t i = 0; i < M; ++i) {
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            // For this (i, p), iterate over all columns of B in chunks of 32
            for (size_t jb = 0; jb < K_ints; ++jb) {
                uint32_t packed = B[p * K_ints + jb];
                size_t j_base = jb * 32;
                // Process all 32 bits
                for (int bit = 0; bit < 32; ++bit) {
                    float sign = (packed & 1) ? 1.0f : -1.0f;
                    C[i * K + j_base + bit] += a_val * sign;
                    packed >>= 1;
                }
            }
        }
    }
}
