#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>

// Final optimized matmul with minimal overhead
// C = A * B where B is a K×K binary matrix packed as 1 bit = ±1.0f

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Initialize C to zero
    memset(C, 0, M * K * sizeof(float));
    
    // Block size for A rows
    const size_t BLOCK_SIZE = 8;
    
    // Process A in blocks of rows
    for (size_t i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
        size_t i1 = (i0 + BLOCK_SIZE <= M) ? i0 + BLOCK_SIZE : M;
        
        // For each row p in B
        for (size_t p = 0; p < K; ++p) {
            // Get the contribution from row p of A for all rows in the block
            float a_vals[8];
            for (size_t i = i0; i < i1; ++i) {
                a_vals[i - i0] = A[i * K + p];
            }
            
            // Process row p of B
            const uint32_t* B_row = B + p * K_ints;
            
            // Process each 32-bit block of B
            for (size_t block = 0; block < K_ints; ++block) {
                uint32_t packed = B_row[block];
                size_t base_j = block * 32;
                
                // Precompute signs for this block
                float signs[32];
                for (int k = 0; k < 32; ++k) {
                    float bit_val = (packed >> k) & 1u;
                    signs[k] = bit_val * 2.0f - 1.0f;
                }
                
                // Accumulate contributions to all rows in the block
                // Process rows one at a time
                for (size_t i = i0; i < i1; ++i) {
                    float* C_row = C + i * K;
                    float a_val = a_vals[i - i0];
                    float* C_row_block = C_row + base_j;
                    float* C_row_block_end = C_row_block + 32;
                    const float* sign_p = signs;
                    
                    // Process columns with pointer arithmetic
                    while (C_row_block < C_row_block_end) {
                        *C_row_block++ += a_val * *sign_p++;
                    }
                }
            }
        }
    }
}
