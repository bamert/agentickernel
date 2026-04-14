// matmul_v8.hpp
#pragma once
#include <cstdint>
#include <cstddef>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// Calculates Matrix C = Matrix A * Matrix B (Optimized Version 8 - SIMD Block Loading)
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        for (size_t j = 0; j < K; j += 32) { // Iterate through C in 32-bit chunks (j is the starting column index)
            
            // Process 32 columns in parallel for C[i, j] to C[i, j+31]
            
            // Load the corresponding block in A: A[i, j] to A[i, j+31]
            float* A_block = &A[i * K + j];

            // For each C column in this block
            for (size_t offset = 0; offset < 32; ++offset) {
                size_t current_j = j + offset;
                float sum = 0.0f;
                
                // Calculate the dot product Sum( A[i, p] * Sign(B[p, current_j]) )
                for (size_t p = 0; p < K; ++p) { 
                    float a_val = A_block[p]; // A[i, current_j + p]
                    
                    // Extract the bit B[p, current_j]
                    size_t packed_idx = p * K_ints + (current_j / 32);
                    uint32_t packed = B[packed_idx];
                    uint32_t bit = (packed >> (current_j % 32)) & 1;
                    
                    float sign = (bit == 1.0f) ? 1.0f : -1.0f;
                    sum += a_val * sign;
                }
                
                C[i * K + current_j] = sum;
            }
        }
    }
}