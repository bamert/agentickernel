#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Use lookup table for sign conversion (faster than conditional)
    const float sign_lut[2] = {-1.0f, 1.0f};
    
    // Allocate aligned memory for better cache performance
    float* sign_matrix = new float[K * K];
    
    // Precompute signs with better bit manipulation
    for (size_t p = 0; p < K; ++p) {
        float* sign_row = sign_matrix + p * K;
        const uint32_t* b_row = B + p * K_ints;
        
        // Process 32 columns per word
        for (size_t word = 0; word < K_ints; ++word) {
            uint32_t bits = b_row[word];
            
            // Process 8 bits at a time using LUT
            for (int byte = 0; byte < 4; ++byte) {
                uint32_t byte_bits = (bits >> (byte * 8)) & 0xFF;
                
                // Unroll to compute all 8 bits for this byte
                for (int bit = 0; bit < 8; ++bit) {
                    size_t col = word * 32 + byte * 8 + bit;
                    size_t bit_idx = (byte_bits >> bit) & 1;
                    sign_row[col] = sign_lut[bit_idx];
                }
            }
        }
    }

    // Improved matrix multiplication with better loop organization
    // Process i, then p, then j for cache efficiency
    for (size_t i = 0; i < M; ++i) {
        // Initialize output row to zero first
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }
        
        // Accumulate: C[i][j] += A[i][p] * B[p][j]
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const float* sign_row = sign_matrix + p * K;
            
            // Unroll the j loop for better performance
            size_t j = 0;
            for (; j + 15 < K; j += 16) {
                C[i * K + j + 0] += a_val * sign_row[j + 0];
                C[i * K + j + 1] += a_val * sign_row[j + 1];
                C[i * K + j + 2] += a_val * sign_row[j + 2];
                C[i * K + j + 3] += a_val * sign_row[j + 3];
                C[i * K + j + 4] += a_val * sign_row[j + 4];
                C[i * K + j + 5] += a_val * sign_row[j + 5];
                C[i * K + j + 6] += a_val * sign_row[j + 6];
                C[i * K + j + 7] += a_val * sign_row[j + 7];
                C[i * K + j + 8] += a_val * sign_row[j + 8];
                C[i * K + j + 9] += a_val * sign_row[j + 9];
                C[i * K + j + 10] += a_val * sign_row[j + 10];
                C[i * K + j + 11] += a_val * sign_row[j + 11];
                C[i * K + j + 12] += a_val * sign_row[j + 12];
                C[i * K + j + 13] += a_val * sign_row[j + 13];
                C[i * K + j + 14] += a_val * sign_row[j + 14];
                C[i * K + j + 15] += a_val * sign_row[j + 15];
            }
            
            // Handle remaining elements
            for (; j < K; ++j) {
                C[i * K + j] += a_val * sign_row[j];
            }
        }
    }

    delete[] sign_matrix;
}
