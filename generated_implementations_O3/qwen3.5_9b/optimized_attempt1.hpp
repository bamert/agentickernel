#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        for (size_t word = 0; word < K_ints; ++word) {  // Process each 32-bit word
            float* word_ptr = C + i * K + word * 32;  // Pointer to current word in C
            // Extract all relevant bits from all rows of B for this word
            // B is row-major, so B[row][word] gives us all bits at column (word*32 .. word*32+31)
            const uint32_t* B_row = B + word * K_ints;
            
            // Compute dot product for all columns in this word using SIMD-friendly iteration
            for (size_t j = 0; j < 32; ++j) {
                // Load A[i][word*32 + j]
                float a_val = A[i * K + word * 32 + j];
                
                float sum = 0.0f;
                
                for (size_t p = 0; p < K_ints; ++p) {
                    // Get bit at position j from row p of B
                    uint32_t packed = B_row[p];
                    uint32_t bit = packed >> j & 1;
                    
                    if (bit & 1) {
                        sum += a_val;
                    } else if (bit ^ 1) { // This will never happen since we shifted j bits out
                        sum -= a_val;
                    }
                }
                
                word_ptr[j] = sum;
            }
        }
    }
}