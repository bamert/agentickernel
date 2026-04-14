#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        const float* rowA = A + i * K;
        float* rowC = C + i * K;
        
        for (size_t j = 0; j < K; ++j) {     // For each column in B
            size_t bitpos = j & 31;
            size_t wordidx = j >> 5;         // j / 32
            float sum = 0.0f;
            
            // For each row p in B
            for (size_t p = 0; p < K_ints; ++p) {
                uint32_t packed = B[p * K_ints + wordidx];
                uint32_t bit = packed >> bitpos & 1u;
                
                // bit=1: +1.0f, bit=0: -1.0f
                sum += rowA[p] * ((bit & 1) ? 1.0f : -1.0f);
            }
            
            rowC[j] = sum;
        }
    }
}