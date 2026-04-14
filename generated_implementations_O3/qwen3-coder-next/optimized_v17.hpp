#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>

// Optimized matmul with improved sign computation (final version)
// C = A * B where B is a K×K binary matrix packed as 1 bit = ±1.0f

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Initialize C to zero
    memset(C, 0, M * K * sizeof(float));
    
    // Process A row by row (no block processing since M=32 fits in cache anyway)
    for (size_t i = 0; i < M; ++i) {
        const float* A_row = A + i * K;
        float* C_row = C + i * K;
        
        for (size_t p = 0; p < K; ++p) {
            float a_val = A_row[p];
            const uint32_t* B_row = B + p * K_ints;
            
            // Process each 32-bit block of B
            for (size_t block = 0; block < K_ints; ++block) {
                uint32_t packed = B_row[block];
                size_t base_j = block * 32;
                
                // Precompute signs for this block using optimized arithmetic
                // (bit_val - 0.5f) * 2.0f = bit? 1.0f : -1.0f
                float signs[32];
                const float factor = 2.0f;
                const float offset = -1.0f;
                for (int k = 0; k < 32; ++k) {
                    signs[k] = ((packed >> k) & 1u) * factor + offset;
                }
                
                // Accumulate contribution to C_row
                float* C_row_block = C_row + base_j;
                
                // Manual unroll
                C_row_block[0]  += a_val * signs[0];
                C_row_block[1]  += a_val * signs[1];
                C_row_block[2]  += a_val * signs[2];
                C_row_block[3]  += a_val * signs[3];
                C_row_block[4]  += a_val * signs[4];
                C_row_block[5]  += a_val * signs[5];
                C_row_block[6]  += a_val * signs[6];
                C_row_block[7]  += a_val * signs[7];
                C_row_block[8]  += a_val * signs[8];
                C_row_block[9]  += a_val * signs[9];
                C_row_block[10] += a_val * signs[10];
                C_row_block[11] += a_val * signs[11];
                C_row_block[12] += a_val * signs[12];
                C_row_block[13] += a_val * signs[13];
                C_row_block[14] += a_val * signs[14];
                C_row_block[15] += a_val * signs[15];
                C_row_block[16] += a_val * signs[16];
                C_row_block[17] += a_val * signs[17];
                C_row_block[18] += a_val * signs[18];
                C_row_block[19] += a_val * signs[19];
                C_row_block[20] += a_val * signs[20];
                C_row_block[21] += a_val * signs[21];
                C_row_block[22] += a_val * signs[22];
                C_row_block[23] += a_val * signs[23];
                C_row_block[24] += a_val * signs[24];
                C_row_block[25] += a_val * signs[25];
                C_row_block[26] += a_val * signs[26];
                C_row_block[27] += a_val * signs[27];
                C_row_block[28] += a_val * signs[28];
                C_row_block[29] += a_val * signs[29];
                C_row_block[30] += a_val * signs[30];
                C_row_block[31] += a_val * signs[31];
            }
        }
    }
}
