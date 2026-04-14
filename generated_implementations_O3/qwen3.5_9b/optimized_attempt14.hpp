#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        const float* rowA = A + i * K;
        float* rowC = C + i * K;
        
        for (size_t j = 0; j < K; ++j) {     // For each column in B
            size_t word = j / 32;
            size_t bitpos = j & 31;
            float sum = 0.0f;
            
            // For each row p in B
            for (size_t p = 0; p < K_ints; ++p) {
                // Get B[p][word]
                uint32_t packed = B[p * K_ints + word];
                // Extract bit at position bitpos
                uint32_t bit = packed >> bitpos & 1u;
                float sign = (bit) ? 1.0f : -1.0f;
                
                sum += rowA[p] * sign;
            }
            
            rowC[j] = sum;
        }
    }
}