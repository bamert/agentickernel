#pragma once
#include <cstdint>
#include <cstddef>

// Optimized with better memory access patterns
// Process 32 output columns at a time to reuse packed B words

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t word_idx = 0; word_idx < K_ints; ++word_idx) {
            // 32 accumulators for the 32 columns in this word
            float accum[32];
            for (int bit = 0; bit < 32; ++bit) {
                accum[bit] = 0.0f;
            }
            
            // Iterate through p (rows of B, elements of A row i)
            for (size_t p = 0; p < K; ++p) {
                float a_val = A[i * K + p];
                uint32_t packed = B[p * K_ints + word_idx];
                
                // Process all 32 bits in parallel
                for (int bit = 0; bit < 32; ++bit) {
                    if ((packed >> bit) & 1) {
                        accum[bit] += a_val;
                    } else {
                        accum[bit] -= a_val;
                    }
                }
            }
            
            // Write results
            for (int bit = 0; bit < 32; ++bit) {
                C[i * K + word_idx * 32 + bit] = accum[bit];
            }
        }
    }
}