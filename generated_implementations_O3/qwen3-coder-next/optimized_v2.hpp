#pragma once
#include <cstdint>
#include <cstddef>

// Optimized matmul: C = A * B
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// K is guaranteed to be a multiple of 32.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Changed loop order to i-p-j for better cache locality
    // For each row i of A
    for (size_t i = 0; i < M; ++i) {
        const float* A_row = A + i * K;
        
        // For each row p of A/B (used to accumulate contributions)
        for (size_t p = 0; p < K; ++p) {
            float a_val = A_row[p];
            
            // Precompute the sign vector for row p of B
            // This row in B contributes a_val * sign[p][j] to each output column j
            for (size_t j = 0; j < K; ++j) {
                uint32_t packed = B[p * K_ints + (j / 32)];
                uint32_t bit = (packed >> (j % 32)) & 1;
                float sign = bit ? 1.0f : -1.0f;
                
                C[i * K + j] += a_val * sign;
            }
        }
    }
}
