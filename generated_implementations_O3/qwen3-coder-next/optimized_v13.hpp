#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>

// Optimized matmul with larger block size for better performance
// C = A * B where B is a K×K binary matrix packed as 1 bit = ±1.0f

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Initialize C to zero
    memset(C, 0, M * K * sizeof(float));
    
    // Block size for A rows - process 8 rows at a time for better cache utilization
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
                    signs[k] = (packed & (1u << k)) ? 1.0f : -1.0f;
                }
                
                // Accumulate contributions to all rows in the block
                for (size_t i = i0; i < i1; ++i) {
                    float* C_row = C + i * K;
                    float a_val = a_vals[i - i0];
                    float* C_row_block = C_row + base_j;
                    
                    // Process 8 columns at a time for better unrolling
                    size_t j = 0;
                    for (; j + 7 < 32; j += 8) {
                        C_row_block[j]   += a_val * signs[j];
                        C_row_block[j+1] += a_val * signs[j+1];
                        C_row_block[j+2] += a_val * signs[j+2];
                        C_row_block[j+3] += a_val * signs[j+3];
                        C_row_block[j+4] += a_val * signs[j+4];
                        C_row_block[j+5] += a_val * signs[j+5];
                        C_row_block[j+6] += a_val * signs[j+6];
                        C_row_block[j+7] += a_val * signs[j+7];
                    }
                    
                    // Handle remaining columns
                    for (; j < 32; ++j) {
                        C_row_block[j] += a_val * signs[j];
                    }
                }
            }
        }
    }
}
