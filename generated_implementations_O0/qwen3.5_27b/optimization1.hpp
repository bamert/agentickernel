#pragma once
#include <cstdint>
#include <cstddef>

// Optimized version with improved memory access pattern
// Uses C = 2 * sum(A[i][p] where B[p][j]==1) - sum(A[i])
// Computes output in blocks of 32 columns for better cache utilization

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Initialize output to zero
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }
    }
    
    // Process each row of C
    for (size_t i = 0; i < M; ++i) {
        const float* A_row = A + i * K;
        float* C_row = C + i * K;
        
        // Process columns in blocks of 32 (aligned with uint32_t packing)
        for (size_t word_idx = 0; word_idx < K_ints; ++word_idx) {
            // Precompute sum of this A row for this output block formula
            float row_sum = 0.0f;
            for (size_t p = 0; p < K; ++p) {
                row_sum += A_row[p];
            }
            
            // Temporary storage for positive contributions in this block
            float contributions[32] = {0};
            
            // For each row in B, check corresponding bits for this word
            for (size_t p = 0; p < K; ++p) {
                uint32_t packed = B[p * K_ints + word_idx];
                
                // Extract each of 32 bits in the word
                for (size_t bit = 0; bit < 32; ++bit) {
                    if ((packed >> bit) & 1) {
                        contributions[bit] += A_row[p];
                    }
                }
            }
            
            // Write final results using formula: 2*pos_sum - row_sum
            for (size_t bit = 0; bit < 32; ++bit) {
                size_t col = word_idx * 32 + bit;
                C_row[col] = contributions[bit] * 2.0f - row_sum;
            }
        }
    }
}
