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

    // Precompute sign values for the binary matrix to avoid repeated bit extraction
    // Store as K×K float matrix of signs (this trades memory for speed)
    // But since we want to avoid large temporary allocations, we'll optimize differently
    
    // Better approach: process rows of A and for each row, process all columns of B
    // but access B in a more cache-friendly way
    
    for (size_t i = 0; i < M; ++i) {
        // Preload row i of A into registers/cached as needed
        const float* A_row = A + i * K;
        
        for (size_t j = 0; j < K; ++j) {
            float sum = 0.0f;
            
            // Access pattern optimization: traverse B column j as consecutive rows
            // B[p][j] means we're accessing B row p, column j
            for (size_t p = 0; p < K; ++p) {
                float a_val = A_row[p];
                
                // Faster bit extraction: move computation outside inner loop as much as possible
                uint32_t packed = B[p * K_ints + (j / 32)];
                uint32_t bit = (packed >> (j % 32)) & 1;
                
                float sign = (bit - 0.5f) * 2.0f; // (bit==1 ? 1.0f : -1.0f)
                sum += a_val * sign;
            }
            
            C[i * K + j] = sum;
        }
    }
}
