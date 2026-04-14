#pragma once
#include <cstdint>
#include <cstddef>

// Optimized version of matmul: same signature, faster bit extraction
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K >> 5;  // K / 32, using shift instead of division

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        for (size_t j = 0; j < K; ++j) {     // For each column in B
            float sum = 0.0f;
            
            // Fast bit extraction: compute word index and bit mask with shifts/masks
            size_t word_idx = j >> 5;           // equivalent to j / 32
            uint32_t mask = (uint32_t)(1u << (j & 31)); // extract the specific bit
            
            for (size_t p = 0; p < K; ++p) {   // Calculate the dot product
                float a_val = A[i * K + p];
                
                // Extract bit from packed word
                uint32_t packed = B[p * K_ints + word_idx];
                float sign = (packed & mask) ? 1.0f : -1.0f;
                sum += a_val * sign;
            }
            
            C[i * K + j] = sum;
        }
    }
}