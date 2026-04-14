#pragma once
#include <cstdint>
#include <cstddef>

// Calculates Matrix C = Matrix A * Matrix B (Optimized v4 - No conditional for sign)
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
// ---------------------------------------------------------
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    const float signs[] = {-1.0f, 1.0f}; // Lookup table for signs

    for (size_t i = 0; i < M; ++i) {               // For each row in A
        for (size_t j = 0; j < K; ++j) {           // Initialize C[i][j] to 0
            C[i * K + j] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) { // For each row in B (and column in A)
            float a_val = A[i * K + p];
            const uint32_t* B_row_ptr = B + p * K_ints; 

            for (size_t k_int = 0; k_int < K_ints; ++k_int) { // For each 32-bit chunk in B's row
                uint32_t packed = B_row_ptr[k_int];
                
                for (size_t bit_idx = 0; bit_idx < 32; ++bit_idx) { // For each bit in the chunk
                    size_t j = k_int * 32 + bit_idx; // Current column index in C
                    float sign = signs[(packed >> bit_idx) & 1]; // Use lookup table
                    C[i * K + j] += a_val * sign;
                }
            }
        }
    }
}