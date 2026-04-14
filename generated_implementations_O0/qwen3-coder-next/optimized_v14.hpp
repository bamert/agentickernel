#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Precompute signs in column-major format for better access during multiplication
    // We want sign_matrix[j * K + p] = sign of B[p][j]
    float* sign_matrix = new float[K * K];
    
    // Precompute signs: convert B to signs column-wise
    for (size_t j = 0; j < K; ++j) {
        float* sign_col = sign_matrix + j * K;
        
        for (size_t p = 0; p < K; ++p) {
            uint32_t packed = B[p * K_ints + (j / 32)];
            uint32_t bit = (packed >> (j % 32)) & 1;
            sign_col[p] = bit ? 1.0f : -1.0f;
        }
    }

    // Matrix multiplication: i, p, j with precomputed column-wise signs
    for (size_t i = 0; i < M; ++i) {
        // Initialize output row to zero first
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }
        
        // Process all p values with j accumulation
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            
            // Access column p of sign_matrix (which is row p in original B)
            const float* sign_col = sign_matrix + p * K;
            
            // Unroll the j loop
            size_t j = 0;
            for (; j + 31 < K; j += 32) {
                C[i * K + j + 0] += a_val * sign_col[j + 0];
                C[i * K + j + 1] += a_val * sign_col[j + 1];
                C[i * K + j + 2] += a_val * sign_col[j + 2];
                C[i * K + j + 3] += a_val * sign_col[j + 3];
                C[i * K + j + 4] += a_val * sign_col[j + 4];
                C[i * K + j + 5] += a_val * sign_col[j + 5];
                C[i * K + j + 6] += a_val * sign_col[j + 6];
                C[i * K + j + 7] += a_val * sign_col[j + 7];
                C[i * K + j + 8] += a_val * sign_col[j + 8];
                C[i * K + j + 9] += a_val * sign_col[j + 9];
                C[i * K + j + 10] += a_val * sign_col[j + 10];
                C[i * K + j + 11] += a_val * sign_col[j + 11];
                C[i * K + j + 12] += a_val * sign_col[j + 12];
                C[i * K + j + 13] += a_val * sign_col[j + 13];
                C[i * K + j + 14] += a_val * sign_col[j + 14];
                C[i * K + j + 15] += a_val * sign_col[j + 15];
                C[i * K + j + 16] += a_val * sign_col[j + 16];
                C[i * K + j + 17] += a_val * sign_col[j + 17];
                C[i * K + j + 18] += a_val * sign_col[j + 18];
                C[i * K + j + 19] += a_val * sign_col[j + 19];
                C[i * K + j + 20] += a_val * sign_col[j + 20];
                C[i * K + j + 21] += a_val * sign_col[j + 21];
                C[i * K + j + 22] += a_val * sign_col[j + 22];
                C[i * K + j + 23] += a_val * sign_col[j + 23];
                C[i * K + j + 24] += a_val * sign_col[j + 24];
                C[i * K + j + 25] += a_val * sign_col[j + 25];
                C[i * K + j + 26] += a_val * sign_col[j + 26];
                C[i * K + j + 27] += a_val * sign_col[j + 27];
                C[i * K + j + 28] += a_val * sign_col[j + 28];
                C[i * K + j + 29] += a_val * sign_col[j + 29];
                C[i * K + j + 30] += a_val * sign_col[j + 30];
                C[i * K + j + 31] += a_val * sign_col[j + 31];
            }
            
            // Handle remaining elements
            for (; j < K; ++j) {
                C[i * K + j] += a_val * sign_col[j];
            }
        }
    }

    delete[] sign_matrix;
}
