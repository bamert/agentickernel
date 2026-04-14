#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        const float* A_row = A + i * K;
        float* C_row = C + i * K;
        
        for (size_t j = 0; j < K; ++j) {     // For each column in B
            float sum = 0.0f;
            
            for (size_t p = 0; p < K_ints; ++p) { // For each word in B
                uint32_t packed = B[p * K_ints + j / 32];
                uint32_t bit = (packed >> (j % 32)) & 1u;
                
                // bit=1 -> sign +1, bit=0 -> sign -1
                sum += A_row[p] * (bit ? 1.0f : -1.0f);
            }
            
            C_row[j] = sum;
        }
        
        // Check if A_row[p] access is correct...
        // A[i*K + p] where p is 0..K-1
        // Yes that's correct
    }
}