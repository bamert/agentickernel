#pragma once
#include <cstdint>
#include <cstddef>

// Calculates Matrix C = Matrix A * Matrix B
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
// ---------------------------------------------------------
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {               // For each row in A
        for (size_t j = 0; j < K; ++j) {           // Initialize C[i][j] to 0
            C[i * K + j] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) { // For each row in B (and column in A)
            float a_val = A[i * K + p];
            const uint32_t* B_row_ptr = B + p * K_ints; 

            for (size_t k_int = 0; k_int < K_ints; ++k_int) { // For each 32-bit chunk in B's row
                uint32_t packed = B_row_ptr[k_int];
                
                // Process positive contributions
                uint32_t current_packed_pos = packed;
                while (current_packed_pos != 0) {
                    uint32_t bit_val = current_packed_pos & (-current_packed_pos); // Isolate the least significant set bit
                    size_t bit_idx = __builtin_ctz(bit_val); // Get index of this bit (count trailing zeros)
                    size_t j = k_int * 32 + bit_idx;
                    C[i * K + j] += a_val;
                    current_packed_pos ^= bit_val; // Clear this bit
                }

                // Process negative contributions: bits that were 0 originally
                uint32_t current_packed_neg = ~packed; // Invert all bits
                // Mask to 32 bits to ensure we only consider relevant bits
                current_packed_neg &= 0xFFFFFFFFU; 
                
                while (current_packed_neg != 0) {
                    uint32_t bit_val = current_packed_neg & (-current_packed_neg);
                    size_t bit_idx = __builtin_ctz(bit_val);
                    size_t j = k_int * 32 + bit_idx;
                    C[i * K + j] -= a_val;
                    current_packed_neg ^= bit_val;
                }
            }
        }
    }
}