#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Optimized: process row i of A, computing contributions to all of C[i]
    for (size_t i = 0; i < M; ++i) {
        const float* rowA = A + i * K;
        float* rowC = C + i * K;
        
        // For each column word (32 columns per word)
        for (size_t word = 0; word < K_ints; ++word) {
            size_t col_mask = word << 5; // word * 32
            const uint32_t* B_word_rows = B + word * K_ints; // Each uint32_t is a row
            
            // For each bit position within the word
            for (size_t bit = 0; bit < 32; ++bit) {
                float a_val = rowA[col_mask + bit];
                float sum = 0.0f;
                
                // For each row in B (contributing to C[i])
                for (size_t r = 0; r < K_ints; ++r) {
                    uint32_t packed = B_word_rows[r];
                    // Check if bit 'bit' is set in packed
                    // This gives us B[row=r, col=word*32+bit]
                    if (packed & (1u << bit)) {
                        sum += a_val; // bit=1 means +1.0f
                    } else {
                        sum -= a_val; // bit=0 means -1.0f
                    }
                }
                
                rowC[col_mask + bit] = sum;
            }
        }
    }
}