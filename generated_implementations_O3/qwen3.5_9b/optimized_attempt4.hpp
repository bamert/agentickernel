#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        for (size_t k = 0; k < K_ints; ++k) { // For each 32-bit column word
            float a_val = A[i * K]; // First element of row i
            // ... this won't work, need to handle each column of A
            
            // Reset and fix
        }
    }
}