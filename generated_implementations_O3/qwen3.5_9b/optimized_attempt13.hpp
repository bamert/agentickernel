#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        const float* A_row = A + i * K;      // Row i of A
        
        for (size_t j = 0; j < K; ++j) {     // For each column in B
            size_t word = j / 32;
            size_t bitpos = j % 32;
            float sum = 0.0f;
            
            // For each row p in B (and thus A)
            for (size_t p = 0; p < K_ints; ++p) {
                uint32_t packed = B[p * K_ints + word];
                uint32_t bit = (packed >> bitpos) & 1u;
                
                sum += A_row[p] * ((bit) ? 1.0f : -1.0f);
            }
            
            C[i * K + j] = sum;
        }
    }
}