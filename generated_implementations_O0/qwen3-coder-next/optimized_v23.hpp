#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Precompute sign matrix in row-major format
    float* sign_matrix = new float[K * K];
    
    // Precompute signs: convert B to signs with better bit manipulation
    for (size_t p = 0; p < K; ++p) {
        float* sign_row = sign_matrix + p * K;
        const uint32_t* b_row = B + p * K_ints;
        
        // Process all 32 bits at once
        for (int bit = 0; bit < 32; ++bit) {
            // Compute signs for bit position across all words
            for (size_t word = 0; word < K_ints; ++word) {
                uint32_t packed = b_row[word];
                uint32_t bit_val = (packed >> bit) & 1;
                sign_row[word * 32 + bit] = bit_val ? 1.0f : -1.0f;
            }
        }
    }

    // Matrix multiplication: i, p, j with optimized memory access
    for (size_t i = 0; i < M; ++i) {
        // Initialize output row to zero first (with unrolling)
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }
        
        // Process all p values with j accumulation
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const float* sign_row = sign_matrix + p * K;
            
            // Unroll the j loop with 32 unrolling
            size_t j = 0;
            for (; j + 31 < K; j += 32) {
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
                C[i * K + j + 16] += a_val * sign_row[j + 16];
                C[i * K + j + 17] += a_val * sign_row[j + 17];
                C[i * K + j + 18] += a_val * sign_row[j + 18];
                C[i * K + j + 19] += a_val * sign_row[j + 19];
                C[i * K + j + 20] += a_val * sign_row[j + 20];
                C[i * K + j + 21] += a_val * sign_row[j + 21];
                C[i * K + j + 22] += a_val * sign_row[j + 22];
                C[i * K + j + 23] += a_val * sign_row[j + 23];
                C[i * K + j + 24] += a_val * sign_row[j + 24];
                C[i * K + j + 25] += a_val * sign_row[j + 25];
                C[i * K + j + 26] += a_val * sign_row[j + 26];
                C[i * K + j + 27] += a_val * sign_row[j + 27];
                C[i * K + j + 28] += a_val * sign_row[j + 28];
                C[i * K + j + 29] += a_val * sign_row[j + 29];
                C[i * K + j + 30] += a_val * sign_row[j + 30];
                C[i * K + j + 31] += a_val * sign_row[j + 31];
            }
            
            // Handle remaining elements
            for (; j < K; ++j) {
                C[i * K + j] += a_val * sign_row[j];
            }
        }
    }

    delete[] sign_matrix;
}
