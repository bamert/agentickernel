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
                
                // Precompute signs for this block using optimized arithmetic
                float signs[32];
                for (int k = 0; k < 32; ++k) {
                    float bit_val = (packed >> k) & 1u;
                    signs[k] = bit_val * 2.0f - 1.0f;
                }
                
                // Accumulate contributions to all rows in the block
                // Process rows one at a time with pointer arithmetic
                for (size_t i = i0; i < i1; ++i) {
                    float* C_row = C + i * K;
                    float a_val = a_vals[i - i0];
                    float* C_row_block = C_row + base_j;
                    const float* sign_p = signs;
                    
                    // Unroll the inner loop
                    C_row_block[0] += a_val * sign_p[0];
                    C_row_block[1] += a_val * sign_p[1];
                    C_row_block[2] += a_val * sign_p[2];
                    C_row_block[3] += a_val * sign_p[3];
                    C_row_block[4] += a_val * sign_p[4];
                    C_row_block[5] += a_val * sign_p[5];
                    C_row_block[6] += a_val * sign_p[6];
                    C_row_block[7] += a_val * sign_p[7];
                    C_row_block[8] += a_val * sign_p[8];
                    C_row_block[9] += a_val * sign_p[9];
                    C_row_block[10] += a_val * sign_p[10];
                    C_row_block[11] += a_val * sign_p[11];
                    C_row_block[12] += a_val * sign_p[12];
                    C_row_block[13] += a_val * sign_p[13];
                    C_row_block[14] += a_val * sign_p[14];
                    C_row_block[15] += a_val * sign_p[15];
                    C_row_block[16] += a_val * sign_p[16];
                    C_row_block[17] += a_val * sign_p[17];
                    C_row_block[18] += a_val * sign_p[18];
                    C_row_block[19] += a_val * sign_p[19];
                    C_row_block[20] += a_val * sign_p[20];
                    C_row_block[21] += a_val * sign_p[21];
                    C_row_block[22] += a_val * sign_p[22];
                    C_row_block[23] += a_val * sign_p[23];
                    C_row_block[24] += a_val * sign_p[24];
                    C_row_block[25] += a_val * sign_p[25];
                    C_row_block[26] += a_val * sign_p[26];
                    C_row_block[27] += a_val * sign_p[27];
                    C_row_block[28] += a_val * sign_p[28];
                    C_row_block[29] += a_val * sign_p[29];
                    C_row_block[30] += a_val * sign_p[30];
                    C_row_block[31] += a_val * sign_p[31];
                }
            }
        }
    }
}
