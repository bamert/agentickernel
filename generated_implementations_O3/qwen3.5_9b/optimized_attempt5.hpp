#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        const float* A_row = A + i * K;      // Pointer to row i of A
        
        for (size_t j = 0; j < K_ints; ++j) { // For each 32-bit word of columns
            const uint32_t* B_col_word = B + j * K_ints; // Each row p has word j
            
            // For each column within the word (0 to 31)
            for (size_t b = 0; b < 32; ++b) {
                float a_val = A_row[j * 32 + b];
                float sum = 0.0f;
                
                // Sum over all rows p
                for (size_t p = 0; p < K_ints; ++p) {
                    uint32_t packed = B_col_word[p];
                    uint32_t bit = (packed >> b) & 0x1;
                    
                    if (bit) {
                        sum += a_val;
                    } else {
                        sum -= a_val;
                    }
                }
                
                C[i * K + j * 32 + b] = sum;
            }
        }
    }
}