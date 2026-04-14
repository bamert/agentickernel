#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        const float* rowA = A + i * K;       // Pointer to row i of A
        
        for (size_t j = 0; j < K; ++j) {     // For each column in B
            size_t word_idx = j / 32;
            size_t bit_idx = j % 32;
            float sum = 0.0f;
            
            // For each row p in B
            for (size_t p = 0; p < K_ints; ++p) {
                // Get the bit at position bit_idx from word word_idx of row p
                uint32_t packed = B[p * K_ints + word_idx];
                // Extract bit and convert to sign (+1 or -1)
                uint32_t bit = packed >> bit_idx & 1u;
                sum += rowA[p] * ((bit) ? 1.0f : -1.0f);
            }
            
            C[i * K + j] = sum;
        }
    }
}