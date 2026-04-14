// matmul_v7.hpp
#pragma once
#include <cstdint>
#include <cstddef>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// Calculates Matrix C = Matrix A * Matrix B (Optimized Version 7 - NEON Vectorization Attempt focusing on outer loops)
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        
        // Process C row by row block of 32 columns (j)
        for (size_t j = 0; j < K; j += 32) {
            
            // Load the relevant 32 columns of A row i: A[i, j] to A[i, j+31]
            // We calculate the dot product for these 32 columns in parallel.
            
            // C block starts at C[i, j]
            float* C_block = &C[i * K + j];

            // Calculate C[i, j] to C[i, j+31]
            for (size_t offset = 0; offset < 32; ++offset) {
                size_t j_start = j + offset;
                
                // Load A values for this C column (A[i, j_start] to A[i, j_start+31])
                // A_vec contains the elements A[i, j_start + p] for p=0..31
                float* A_vec = &A[i * K + j_start];
                
                // --- SIMD calculation for the dot product (j_start) ------------
                
                // Initialize sum for C[i, j_start]
                float sum = 0.0f;
                
                // Iterate over the rows p of B (K iterations) to calculate the dot product.
                for (size_t p = 0; p < K; ++p) { 
                    float a_val = A_vec[p]; // A[i, j_start + p]
                    
                    // Extract the bit B[p, j_start]
                    size_t packed_idx = p * K_ints + (j_start / 32);
                    uint32_t packed = B[packed_idx];
                    uint32_t bit = (packed >> (j_start % 32)) & 1;
                    
                    float sign = (bit == 1.0f) ? 1.0f : -1.0f;
                    sum += a_val * sign;
                }
                
                C_block[offset] = sum;
            }
        }
    }
}