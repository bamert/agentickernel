#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Precompute signs in row-major format but inline more efficiently
    float* sign_matrix = new float[K * K];
    
    // Precompute signs: convert B to signs with better bit manipulation
    for (size_t p = 0; p < K; ++p) {
        float* sign_row = sign_matrix + p * K;
        const uint32_t* b_row = B + p * K_ints;
        
        // Unroll the word loop
        for (size_t word = 0; word < K_ints; word += 4) {
            for (int w = 0; w < 4 && (word + w) < K_ints; ++w) {
                uint32_t bits = b_row[word + w];
                
                // Unroll 8 bits at a time
                for (int bit = 0; bit < 32; bit += 8) {
                    uint32_t byte_bits = (bits >> bit) & 0xFF;
                    
                    // Process all 8 bits in the byte
                    sign_row[word * 32 + word * 8 + bit + 0] = ((byte_bits >> 0) & 1) ? 1.0f : -1.0f;
                    sign_row[word * 32 + word * 8 + bit + 1] = ((byte_bits >> 1) & 1) ? 1.0f : -1.0f;
                    sign_row[word * 32 + word * 8 + bit + 2] = ((byte_bits >> 2) & 1) ? 1.0f : -1.0f;
                    sign_row[word * 32 + word * 8 + bit + 3] = ((byte_bits >> 3) & 1) ? 1.0f : -1.0f;
                    sign_row[word * 32 + word * 8 + bit + 4] = ((byte_bits >> 4) & 1) ? 1.0f : -1.0f;
                    sign_row[word * 32 + word * 8 + bit + 5] = ((byte_bits >> 5) & 1) ? 1.0f : -1.0f;
                    sign_row[word * 32 + word * 8 + bit + 6] = ((byte_bits >> 6) & 1) ? 1.0f : -1.0f;
                    sign_row[word * 32 + word * 8 + bit + 7] = ((byte_bits >> 7) & 1) ? 1.0f : -1.0f;
                }
            }
        }
    }

    // Matrix multiplication: i, p, j order with better unrolling
    for (size_t i = 0; i < M; ++i) {
        // Initialize output row to zero
        size_t j = 0;
        for (; j + 7 < K; j += 8) {
            C[i * K + j + 0] = 0.0f;
            C[i * K + j + 1] = 0.0f;
            C[i * K + j + 2] = 0.0f;
            C[i * K + j + 3] = 0.0f;
            C[i * K + j + 4] = 0.0f;
            C[i * K + j + 5] = 0.0f;
            C[i * K + j + 6] = 0.0f;
            C[i * K + j + 7] = 0.0f;
        }
        for (; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }
        
        // Accumulate: C[i][j] += A[i][p] * B[p][j]
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const float* sign_row = sign_matrix + p * K;
            
            // Unroll inner loop
            j = 0;
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
            
            // Handle remaining elements
            for (; j < K; ++j) {
                C[i * K + j] += a_val * sign_row[j];
            }
        }
    }

    delete[] sign_matrix;
}
