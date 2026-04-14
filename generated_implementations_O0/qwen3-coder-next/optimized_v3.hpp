#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Precompute sign matrix as floats: K rows × K cols
    // Store in row-major: sign_row_p[j] = sign of B[p][j]
    alignas(32) float* sign_matrix = new float[K * K];
    
    for (size_t p = 0; p < K; ++p) {
        // Precompute signs for this row of B
        float* sign_row = sign_matrix + p * K;
        for (size_t word_idx = 0; word_idx < K_ints; ++word_idx) {
            uint32_t packed_word = B[p * K_ints + word_idx];
            
            // Process 32 bits at a time
            for (int bit_pos = 0; bit_pos < 32; ++bit_pos) {
                size_t col = word_idx * 32 + bit_pos;
                uint32_t bit = (packed_word >> bit_pos) & 1;
                sign_row[col] = bit ? 1.0f : -1.0f;
            }
        }
    }

    // Perform matrix multiplication
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < K; ++p) {
                sum += A[i * K + p] * sign_matrix[p * K + j];
            }
            C[i * K + j] = sum;
        }
    }

    delete[] sign_matrix;
}
