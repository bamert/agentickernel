// matmul_v1.hpp
#pragma once
#include <cstdint>
#include <cstddef>

// Calculates Matrix C = Matrix A * Matrix B (Optimized Version 1 - Structure focused)
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        for (size_t j = 0; j < K; ++j) {     // For each column in B (C[i, j])
            float sum = 0.0f;
            
            // Calculate the dot product: Sum( A[i, p] * Sign(B[p, j]) )
            for (size_t p = 0; p < K; ++p) { 
                float a_val = A[i * K + p];
                
                // Extract the specific bit at B[p][j]
                // B is row-major: index = p * K_ints + (j / 32) for the 32-bit word
                uint32_t packed = B[p * K_ints + (j / 32)];
                uint32_t bit = (packed >> (j % 32)) & 1;
                
                float sign = bit ? 1.0f : -1.0f;
                sum += a_val * sign;
            }
            
            C[i * K + j] = sum;
        }
    }
}