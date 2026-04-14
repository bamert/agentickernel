#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Allocate sign matrix but organize it column-wise (transposed)
    // This allows better cache access when summing across p for fixed j
    float* sign_matrix = new float[K * K];
    
    // Precompute signs in row-major (p, j) format
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

    // Changed loop order: i, j, p - this might be better for C output
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            float sum = 0.0f;
            
            // Process p loop with good locality for sign_row
            for (size_t p = 0; p < K; ++p) {
                float a_val = A[i * K + p];
                float sign_val = sign_matrix[p * K + j];
                sum += a_val * sign_val;
            }
            
            C[i * K + j] = sum;
        }
    }

    delete[] sign_matrix;
}
