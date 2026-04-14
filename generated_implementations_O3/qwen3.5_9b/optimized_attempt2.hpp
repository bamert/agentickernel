#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        for (size_t p = 0; p < K_ints; ++p) { // For each word in B
            const uint32_t* B_row = B + p * K_ints;
            float* C_row = C + i * K + p * 32; // Start addressing C for word p
            
            // Unroll the column iteration for each bit position
            for (size_t j = 0; j < 32; ++j) {
                float a_val = A[i * K + p * 32 + j];
                
                // Process all bits at position j across all rows for this word
                for (size_t bit = 0; bit < 32; ++bit) {
                    uint32_t packed = B_row[j + bit];
                    uint32_t col = j; // This is wrong - need to track which column we're computing
                    // Actually, we need to compute C[i*K + bit + j] for bit = 0..31
                    // So for each column k = p*32 + bit (relative to row p)
                    // B_bit_index = p * K_ints + col / 32
                    // B_bit_offset = col % 32
                }
            }
        }
    }
}