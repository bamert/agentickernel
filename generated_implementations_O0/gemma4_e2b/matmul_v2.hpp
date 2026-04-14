// matmul_v2.hpp
#pragma once
#include <cstdint>
#include <cstddef>

// Calculates Matrix C = Matrix A * Matrix B (Optimized Version 2 - Loop Reordering focus)
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        for (size_t p = 0; p < K; ++p) {     // Iterate over rows of B (p is the index for the row we are reading from B)
            float a_val = A[i * K + p];
            
            // Pre-calculate the entire required row of signs derived from B[p, :]
            // This is still inefficient as we calculate C[i, j] iteratively.
            // We return to the original structure but focus on optimizing the inner term extraction.

            for (size_t j = 0; j < K; ++j) {     // For each column j in C
                float sum = 0.0f;
                
                // Calculate the dot product: Sum( A[i, p] * Sign(B[p, j]) )
                for (size_t q = 0; q < K; ++q) { // p acts as the iterating index here (renamed to q for clarity in dot product)
                    float a_val_p = A[i * K + q]; // A[i, q]
                    
                    // Extract the specific bit at B[q][j] (B[row=q, col=j])
                    // B is stored row-major, where each row is packed.
                    // The index for the 32-bit word containing row q's data, and the bit offset j.
                    uint32_t packed = B[q * K_ints + (j / 32)];
                    uint32_t bit = (packed >> (j % 32)) & 1;
                    
                    float sign = bit ? 1.0f : -1.0f;
                    sum += a_val_p * sign;
                }
                
                C[i * K + j] = sum;
            }
        }
    }
}