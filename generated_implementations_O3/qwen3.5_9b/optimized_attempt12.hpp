#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        const float* A_row = A + i * K;      // Pointer to row i of A
        float* C_row = C + i * K;            // Pointer to row i of C
        
        // For each column j
        for (size_t j = 0; j < K; ++j) {     // For each column in B
            size_t word = j / 32;
            size_t bitpos = j % 32;
            float sum = 0.0f;
            
            // For each row p in B
            for (size_t p = 0; p < K_ints; ++p) {
                // B is row-major: B[p * K_ints + word] gets row p, word containing column j
                uint32_t packed = B[p * K_ints + word];
                // Extract bit at bitpos
                uint32_t bit = (packed >> bitpos) & 1;
                
                // bit=1: +A[p], bit=0: -A[p]
                sum += A_row[p] * ((bit & 1) ? 1.0f : -1.0f);
            }
            
            C_row[j] = sum;
        }
    }
}