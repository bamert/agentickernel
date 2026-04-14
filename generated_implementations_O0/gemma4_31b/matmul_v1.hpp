#pragma once
#include <cstdint>
#include <cstddef>

// Optimized Matrix Multiplication with Packed Binary Matrix B
// Logic: C[i][j] = sum_{p=0..K-1} A[i][p] * (B[p][j] ? 1 : -1)
// Transformation: C[i][j] = 2 * sum_{p: B[p][j]=1} A[i][p] - sum_{p=0..K-1} A[i][p]
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* row_A = A + i * K;
        float* row_C = C + i * K;

        // 1. Precompute the total sum of row A_i
        float sum_A = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            sum_A += row_A[p];
        }

        // 2. Initialize C_ij with -sum_A
        // This handles the -1.0f part of the sign for all elements
        for (size_t j = 0; j < K; ++j) {
            row_C[j] = -sum_A;
        }

        // 3. Add 2 * A[i][p] for every bit set in B[p][j]
        // Loop order: i -> p -> b to ensure sequential access to B and use L1 for row_C
        for (size_t p = 0; p < K; ++p) {
            const float a_val_2 = 2.0f * row_A[p];
            if (a_val_2 == 0.0f) continue;

            const uint32_t* row_B_packed = B + p * K_ints;
            
            for (size_t b = 0; b < K_ints; ++b) {
                uint32_t packed = row_B_packed[b];
                if (packed == 0) continue;

                const size_t offset = b * 32;
                
                // Extract set bits efficiently
                while (packed != 0) {
                    // __builtin_ctz finds the index of the trailing zero bits
                    int bit = __builtin_ctz(packed);
                    row_C[offset + bit] += a_val_2;
                    // Clear the bit we just processed
                    packed &= ~(1u << bit);
                }
            }
        }
    }
}
