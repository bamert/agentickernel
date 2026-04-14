#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        for (size_t k = 0; k < K_ints; ++k) { // For each 32-bit word of columns
            const uint32_t* A_row = A + i * K; // Pointer to row i of A
            
            // For each column word, we compute 32 columns of C
            for (size_t j = 0; j < 32; ++j) {
                float a_val = A_row[k * 32 + j];
                float sum = 0.0f;
                
                // For each bit position within the word (representing different bit offsets)
                for (size_t b = 0; b < K_ints; ++b) {
                    // Get the packed value for this column word in B
                    uint32_t packed = B[b * K_ints + k];
                    // Check if bit j is set
                    unsigned int bit = packed >> j & 1;
                    // bit=1 means +1.0f, bit=0 means -1.0f
                    sum += a_val * ((bit) ? 1.0f : -1.0f);
                }
                
                C[i * K + k * 32 + j] = sum;
            }
        }
    }
}