#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        const float* rowA = A + i * K;
        float* rowC = C + i * K;
        
        for (size_t p = 0; p < K_ints; ++p) { // For each 32-bit word in B (representing 32 columns)
            const uint32_t* rowB = B + p * K_ints; // Pointer to u32s for row p of B (all word columns)
            
            // For each bit position within the word (0-31)
            for (size_t b = 0; b < 32; ++b) {
                float sum = 0.0f;
                uint32_t mask = 0x80000000u >> b; // Create mask for bit position b
                
                // For each row in B (and column in A)
                for (size_t r = 0; r < K_ints; ++r) {
                    // Check if bit b is set in u32 r of row r of B
                    // This represents B[r][p*32 + b]
                    if ((rowB[r] >> b) & 1u) {
                        sum += rowA[p * 32 + b];
                    } else {
                        sum -= rowA[p * 32 + b];
                    }
                }
                
                rowC[p * 32 + b] = sum;
            }
        }
    }
}