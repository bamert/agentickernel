#pragma once
#include <cstdint>
#include <cstddef>

// Optimized with better loop structure and reduced redundant bit operations
// Process each row of A, and for each word of B, extract all 32 bits at once

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    for (size_t i = 0; i < M; ++i) {
        const float* A_row = A + i * K;
        float* C_row = C + i * K;
        
        // For each output column block (32 columns at a time)
        for (size_t word = 0; word < K_ints; ++word) {
            // Temporary accumulators for each of 32 output columns in this word
            float accum[32] = {0};
            
            // For each row in B (which corresponds to each element in A_row)
            for (size_t p = 0; p < K; ++p) {
                float a_val = A_row[p];
                uint32_t packed = B[p * K_ints + word];
                
                // Process all 32 bits from this word
                for (int bit = 0; bit < 32; ++bit) {
                    uint32_t mask = (uint32_t)1 << bit;
                    if (packed & mask) {
                        accum[bit] += a_val;   // bit = 1, multiply by +1
                    } else {
                        accum[bit] -= a_val;   // bit = 0, multiply by -1
                    }
                }
            }
            
            // Write 32 results at once
            for (int bit = 0; bit < 32; ++bit) {
                C_row[word * 32 + bit] = accum[bit];
            }
        }
    }
}