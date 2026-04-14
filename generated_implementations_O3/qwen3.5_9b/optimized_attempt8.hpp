#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        const float* A_row = A + i * K;
        float* C_row = C + i * K;
        
        // For each column j
        for (size_t j = 0; j < K; ++j) {
            size_t j_word = j >> 5;   // j / 32
            size_t j_bit = j & 31;    // j % 32
            float sum = 0.0f;
            
            // For each row p in B (dot product with row i of A)
            for (size_t p = 0; p < K_ints; ++p) {
                const uint32_t* B_row = B + p * K_ints;
                // Access row p, column j_word from B
                uint32_t packed = B_row[j_word];
                // Extract bit j_bit
                uint32_t bit = (packed >> j_bit) & 1u;
                
                float sign = (bit) ? 1.0f : -1.0f;
                sum += A_row[j] * sign;
            }
            
            C_row[j] = sum;
        }
    }
}