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
        
        for (size_t word = 0; word < K_ints; ++word) {
            uint32_t bits = b_row[word];
            
            // Compute 32 signs from this word with unrolled bit shifts
            sign_row[word * 32 + 0] = ((bits >> 0) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 1] = ((bits >> 1) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 2] = ((bits >> 2) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 3] = ((bits >> 3) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 4] = ((bits >> 4) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 5] = ((bits >> 5) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 6] = ((bits >> 6) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 7] = ((bits >> 7) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 8] = ((bits >> 8) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 9] = ((bits >> 9) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 10] = ((bits >> 10) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 11] = ((bits >> 11) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 12] = ((bits >> 12) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 13] = ((bits >> 13) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 14] = ((bits >> 14) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 15] = ((bits >> 15) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 16] = ((bits >> 16) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 17] = ((bits >> 17) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 18] = ((bits >> 18) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 19] = ((bits >> 19) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 20] = ((bits >> 20) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 21] = ((bits >> 21) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 22] = ((bits >> 22) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 23] = ((bits >> 23) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 24] = ((bits >> 24) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 25] = ((bits >> 25) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 26] = ((bits >> 26) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 27] = ((bits >> 27) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 28] = ((bits >> 28) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 29] = ((bits >> 29) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 30] = ((bits >> 30) & 1) ? 1.0f : -1.0f;
            sign_row[word * 32 + 31] = ((bits >> 31) & 1) ? 1.0f : -1.0f;
        }
    }

    // Matrix multiplication: i, p, j with optimized memory access
    for (size_t i = 0; i < M; ++i) {
        // Initialize output row to zero first (with unrolling)
        size_t j = 0;
        for (; j + 31 < K; j += 32) {
            C[i * K + j + 0] = 0.0f;
            C[i * K + j + 1] = 0.0f;
            C[i * K + j + 2] = 0.0f;
            C[i * K + j + 3] = 0.0f;
            C[i * K + j + 4] = 0.0f;
            C[i * K + j + 5] = 0.0f;
            C[i * K + j + 6] = 0.0f;
            C[i * K + j + 7] = 0.0f;
            C[i * K + j + 8] = 0.0f;
            C[i * K + j + 9] = 0.0f;
            C[i * K + j + 10] = 0.0f;
            C[i * K + j + 11] = 0.0f;
            C[i * K + j + 12] = 0.0f;
            C[i * K + j + 13] = 0.0f;
            C[i * K + j + 14] = 0.0f;
            C[i * K + j + 15] = 0.0f;
            C[i * K + j + 16] = 0.0f;
            C[i * K + j + 17] = 0.0f;
            C[i * K + j + 18] = 0.0f;
            C[i * K + j + 19] = 0.0f;
            C[i * K + j + 20] = 0.0f;
            C[i * K + j + 21] = 0.0f;
            C[i * K + j + 22] = 0.0f;
            C[i * K + j + 23] = 0.0f;
            C[i * K + j + 24] = 0.0f;
            C[i * K + j + 25] = 0.0f;
            C[i * K + j + 26] = 0.0f;
            C[i * K + j + 27] = 0.0f;
            C[i * K + j + 28] = 0.0f;
            C[i * K + j + 29] = 0.0f;
            C[i * K + j + 30] = 0.0f;
            C[i * K + j + 31] = 0.0f;
        }
        for (; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }
        
        // Process all p values with j accumulation
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const float* sign_row = sign_matrix + p * K;
            
            // Unroll the j loop with 32 unrolling
            j = 0;
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
