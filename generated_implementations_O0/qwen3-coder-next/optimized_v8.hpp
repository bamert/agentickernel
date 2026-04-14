#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Precompute sign matrix in row-major format
    float* sign_matrix = new float[K * K];
    
    // Precompute signs: convert B to signs efficiently
    for (size_t p = 0; p < K; ++p) {
        float* sign_row = sign_matrix + p * K;
        const uint32_t* b_row = B + p * K_ints;
        
        for (size_t word = 0; word < K_ints; ++word) {
            uint32_t bits = b_row[word];
            
            for (int bit = 0; bit < 32; ++bit) {
                size_t col = word * 32 + bit;
                sign_row[col] = ((bits >> bit) & 1) ? 1.0f : -1.0f;
            }
        }
    }

    // Matrix multiplication with blocked loops for better cache usage
    const int BLOCK_SIZE = 128;
    
    for (size_t i = 0; i < M; ++i) {
        // Initialize output row to zero first
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }
        
        // Process B in blocks for better cache utilization
        for (size_t p_start = 0; p_start < K; p_start += BLOCK_SIZE) {
            size_t p_end = std::min(p_start + BLOCK_SIZE, K);
            
            for (size_t p = p_start; p < p_end; ++p) {
                float a_val = A[i * K + p];
                const float* sign_row = sign_matrix + p * K;
                
                // Unroll the j loop
                size_t j = 0;
                for (; j + 7 < K; j += 8) {
                    C[i * K + j + 0] += a_val * sign_row[j + 0];
                    C[i * K + j + 1] += a_val * sign_row[j + 1];
                    C[i * K + j + 2] += a_val * sign_row[j + 2];
                    C[i * K + j + 3] += a_val * sign_row[j + 3];
                    C[i * K + j + 4] += a_val * sign_row[j + 4];
                    C[i * K + j + 5] += a_val * sign_row[j + 5];
                    C[i * K + j + 6] += a_val * sign_row[j + 6];
                    C[i * K + j + 7] += a_val * sign_row[j + 7];
                }
                
                for (; j < K; ++j) {
                    C[i * K + j] += a_val * sign_row[j];
                }
            }
        }
    }

    delete[] sign_matrix;
}
