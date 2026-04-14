#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Precompute sign matrix in row-major format
    float* sign_matrix = new float[K * K];
    
    // Precompute signs: convert B to signs with better bit manipulation
    // Process all p values first, then compute signs for all j values
    for (size_t p = 0; p < K; ++p) {
        float* sign_row = sign_matrix + p * K;
        const uint32_t* b_row = B + p * K_ints;
        
        // Precompute all signs for this row
        for (size_t word = 0; word < K_ints; ++word) {
            uint32_t bits = b_row[word];
            
            // Compute 32 signs from this word
            for (int bit = 0; bit < 32; ++bit) {
                sign_row[word * 32 + bit] = ((bits >> bit) & 1) ? 1.0f : -1.0f;
            }
        }
    }

    // Matrix multiplication: i, p, j with optimized memory access and better unrolling
    for (size_t i = 0; i < M; ++i) {
        // Initialize output row to zero first (with unrolling)
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }
        
        // Process all p values with j accumulation
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const float* sign_row = sign_matrix + p * K;
            
            // Unroll the j loop
            size_t j = 0;
            for (; j + 31 < K; j += 32) {
                // 32 unrolling with pointer arithmetic for better performance
                float& c0 = C[i * K + j + 0];
                float& c1 = C[i * K + j + 1];
                float& c2 = C[i * K + j + 2];
                float& c3 = C[i * K + j + 3];
                float& c4 = C[i * K + j + 4];
                float& c5 = C[i * K + j + 5];
                float& c6 = C[i * K + j + 6];
                float& c7 = C[i * K + j + 7];
                float& c8 = C[i * K + j + 8];
                float& c9 = C[i * K + j + 9];
                float& c10 = C[i * K + j + 10];
                float& c11 = C[i * K + j + 11];
                float& c12 = C[i * K + j + 12];
                float& c13 = C[i * K + j + 13];
                float& c14 = C[i * K + j + 14];
                float& c15 = C[i * K + j + 15];
                float& c16 = C[i * K + j + 16];
                float& c17 = C[i * K + j + 17];
                float& c18 = C[i * K + j + 18];
                float& c19 = C[i * K + j + 19];
                float& c20 = C[i * K + j + 20];
                float& c21 = C[i * K + j + 21];
                float& c22 = C[i * K + j + 22];
                float& c23 = C[i * K + j + 23];
                float& c24 = C[i * K + j + 24];
                float& c25 = C[i * K + j + 25];
                float& c26 = C[i * K + j + 26];
                float& c27 = C[i * K + j + 27];
                float& c28 = C[i * K + j + 28];
                float& c29 = C[i * K + j + 29];
                float& c30 = C[i * K + j + 30];
                float& c31 = C[i * K + j + 31];
                
                c0 += a_val * sign_row[j + 0];
                c1 += a_val * sign_row[j + 1];
                c2 += a_val * sign_row[j + 2];
                c3 += a_val * sign_row[j + 3];
                c4 += a_val * sign_row[j + 4];
                c5 += a_val * sign_row[j + 5];
                c6 += a_val * sign_row[j + 6];
                c7 += a_val * sign_row[j + 7];
                c8 += a_val * sign_row[j + 8];
                c9 += a_val * sign_row[j + 9];
                c10 += a_val * sign_row[j + 10];
                c11 += a_val * sign_row[j + 11];
                c12 += a_val * sign_row[j + 12];
                c13 += a_val * sign_row[j + 13];
                c14 += a_val * sign_row[j + 14];
                c15 += a_val * sign_row[j + 15];
                c16 += a_val * sign_row[j + 16];
                c17 += a_val * sign_row[j + 17];
                c18 += a_val * sign_row[j + 18];
                c19 += a_val * sign_row[j + 19];
                c20 += a_val * sign_row[j + 20];
                c21 += a_val * sign_row[j + 21];
                c22 += a_val * sign_row[j + 22];
                c23 += a_val * sign_row[j + 23];
                c24 += a_val * sign_row[j + 24];
                c25 += a_val * sign_row[j + 25];
                c26 += a_val * sign_row[j + 26];
                c27 += a_val * sign_row[j + 27];
                c28 += a_val * sign_row[j + 28];
                c29 += a_val * sign_row[j + 29];
                c30 += a_val * sign_row[j + 30];
                c31 += a_val * sign_row[j + 31];
            }
            
            // Handle remaining elements
            for (; j < K; ++j) {
                C[i * K + j] += a_val * sign_row[j];
            }
        }
    }

    delete[] sign_matrix;
}
