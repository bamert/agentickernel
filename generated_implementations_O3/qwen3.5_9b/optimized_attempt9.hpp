#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        const float* A_row = A + i * K;
        float* C_row = C + i * K;
        
        // For each column word in B (32 columns per word)
        for (size_t word = 0; word < K_ints; ++word) {
            // For each row p in B
            for (size_t p = 0; p < K_ints; ++p) {
                // For each column within the word (0-31)
                for (size_t bit = 0; bit < 32; ++bit) {
                    // Get the bit from B at row p, column word
                    uint32_t packed = B[p * K_ints + word];
                    uint32_t bit_val = (packed >> bit) & 1u;
                    
                    // If bit_val is 1, we add A_row[bit]; if 0, we subtract A_row[bit]
                    float a_val = A_row[bit]; // A is 8-bit? No, A floats...
                    // Wait, A_row[bit] is just the first column of row i at column 'bit'
                    // We need A_row[bit] for each bit position
                    
                    // But wait, A is row-major A[i*K + p] for column p
                    // When computing C[i][bit], we need A[i][bit], not A[i][p]
                    // This is wrong.
                    
                    // Let me redo this properly:
                    // C[i][j] = sum_{p=0}^{K_ints-1} A[i][p] * sign(B[p][j])
                    // So for column j (where j = bit within word), we iterate p, and use A[i][p]
                    // But we iterate bit for columns, so we need A for each bit
                
                    // Start fresh with correct logic
                }
            }
        }
    }
}