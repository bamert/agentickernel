#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>

// Optimized matmul with improved access patterns
// C = A * B where B is a K×K binary matrix packed as 1 bit = ±1.0f

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Initialize C to zero
    memset(C, 0, M * K * sizeof(float));
    
    // Process rows of A and columns of B
    for (size_t i = 0; i < M; ++i) {
        const float* A_row = A + i * K;
        float* C_row = C + i * K;
        
        // For each row p in both A and B
        for (size_t p = 0; p < K; ++p) {
            float a_val = A_row[p];
            
            // Process B row in large chunks
            const uint32_t* B_row = B + p * K_ints;
            
            // Accumulate contributions to C_row
            // Process 4 columns at a time to reduce loop overhead
            for (size_t block = 0; block < K_ints; ++block) {
                uint32_t packed = B_row[block];
                size_t base_j = block * 32;
                
                // Unroll the 32 iterations
                #define COMPUTE_SIGN(j) ((packed >> (j)) & 1u ? 1.0f : -1.0f)
                
                C_row[base_j + 0]  += a_val * COMPUTE_SIGN(0);
                C_row[base_j + 1]  += a_val * COMPUTE_SIGN(1);
                C_row[base_j + 2]  += a_val * COMPUTE_SIGN(2);
                C_row[base_j + 3]  += a_val * COMPUTE_SIGN(3);
                C_row[base_j + 4]  += a_val * COMPUTE_SIGN(4);
                C_row[base_j + 5]  += a_val * COMPUTE_SIGN(5);
                C_row[base_j + 6]  += a_val * COMPUTE_SIGN(6);
                C_row[base_j + 7]  += a_val * COMPUTE_SIGN(7);
                C_row[base_j + 8]  += a_val * COMPUTE_SIGN(8);
                C_row[base_j + 9]  += a_val * COMPUTE_SIGN(9);
                C_row[base_j + 10] += a_val * COMPUTE_SIGN(10);
                C_row[base_j + 11] += a_val * COMPUTE_SIGN(11);
                C_row[base_j + 12] += a_val * COMPUTE_SIGN(12);
                C_row[base_j + 13] += a_val * COMPUTE_SIGN(13);
                C_row[base_j + 14] += a_val * COMPUTE_SIGN(14);
                C_row[base_j + 15] += a_val * COMPUTE_SIGN(15);
                C_row[base_j + 16] += a_val * COMPUTE_SIGN(16);
                C_row[base_j + 17] += a_val * COMPUTE_SIGN(17);
                C_row[base_j + 18] += a_val * COMPUTE_SIGN(18);
                C_row[base_j + 19] += a_val * COMPUTE_SIGN(19);
                C_row[base_j + 20] += a_val * COMPUTE_SIGN(20);
                C_row[base_j + 21] += a_val * COMPUTE_SIGN(21);
                C_row[base_j + 22] += a_val * COMPUTE_SIGN(22);
                C_row[base_j + 23] += a_val * COMPUTE_SIGN(23);
                C_row[base_j + 24] += a_val * COMPUTE_SIGN(24);
                C_row[base_j + 25] += a_val * COMPUTE_SIGN(25);
                C_row[base_j + 26] += a_val * COMPUTE_SIGN(26);
                C_row[base_j + 27] += a_val * COMPUTE_SIGN(27);
                C_row[base_j + 28] += a_val * COMPUTE_SIGN(28);
                C_row[base_j + 29] += a_val * COMPUTE_SIGN(29);
                C_row[base_j + 30] += a_val * COMPUTE_SIGN(30);
                C_row[base_j + 31] += a_val * COMPUTE_SIGN(31);
                
                #undef COMPUTE_SIGN
            }
        }
    }
}
