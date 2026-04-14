#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Precompute the sign matrix in a more cache-friendly way
    // Allocate aligned memory for better cache performance
    float* sign_matrix = new float[K * K];
    
    // Process B row by row, converting bits to signs efficiently
    for (size_t p = 0; p < K; ++p) {
        float* sign_row = sign_matrix + p * K;
        const uint32_t* b_row = B + p * K_ints;
        
        for (size_t word = 0; word < K_ints; ++word) {
            uint32_t bits = b_row[word];
            
            // Compute signs for all 32 bits in the word
            for (int bit = 0; bit < 32; ++bit) {
                size_t col = word * 32 + bit;
                sign_row[col] = ((bits >> bit) & 1) ? 1.0f : -1.0f;
            }
        }
    }

    // Matrix multiplication: C[i][j] = sum_p A[i][p] * sign_matrix[p][j]
    // Reorder loops to be cache-friendly (i, p, j)
    for (size_t i = 0; i < M; ++i) {
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const float* sign_row = sign_matrix + p * K;
            
            // Vectorize the accumulation over columns
            for (size_t j = 0; j < K; ++j) {
                C[i * K + j] += a_val * sign_row[j];
            }
        }
    }

    delete[] sign_matrix;
}
