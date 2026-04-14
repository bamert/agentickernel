#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>

// Optimized matmul: C = A * B
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// K is guaranteed to be a multiple of 32.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Initialize C to zero
    memset(C, 0, M * K * sizeof(float));
    
    // Block size for A rows - tuned for cache efficiency
    const size_t BLOCK_SIZE = 8;
    
    // Process A in blocks of rows for better cache locality
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
                
                // Precompute signs for this block: bit_val * 2.0f - 1.0f gives ±1.0f
                float signs[32];
                for (int k = 0; k < 32; ++k) {
                    float bit_val = (packed >> k) & 1u;
                    signs[k] = bit_val * 2.0f - 1.0f;
                }
                
                // Accumulate contributions to all rows in the block
                for (size_t i = i0; i < i1; ++i) {
                    float a_val = a_vals[i - i0];
                    float* C_row = C + i * K + base_j;
                    
                    // Manual unroll of 32-column loop for better performance
                    C_row[0]  += a_val * signs[0];
                    C_row[1]  += a_val * signs[1];
                    C_row[2]  += a_val * signs[2];
                    C_row[3]  += a_val * signs[3];
                    C_row[4]  += a_val * signs[4];
                    C_row[5]  += a_val * signs[5];
                    C_row[6]  += a_val * signs[6];
                    C_row[7]  += a_val * signs[7];
                    C_row[8]  += a_val * signs[8];
                    C_row[9]  += a_val * signs[9];
                    C_row[10] += a_val * signs[10];
                    C_row[11] += a_val * signs[11];
                    C_row[12] += a_val * signs[12];
                    C_row[13] += a_val * signs[13];
                    C_row[14] += a_val * signs[14];
                    C_row[15] += a_val * signs[15];
                    C_row[16] += a_val * signs[16];
                    C_row[17] += a_val * signs[17];
                    C_row[18] += a_val * signs[18];
                    C_row[19] += a_val * signs[19];
                    C_row[20] += a_val * signs[20];
                    C_row[21] += a_val * signs[21];
                    C_row[22] += a_val * signs[22];
                    C_row[23] += a_val * signs[23];
                    C_row[24] += a_val * signs[24];
                    C_row[25] += a_val * signs[25];
                    C_row[26] += a_val * signs[26];
                    C_row[27] += a_val * signs[27];
                    C_row[28] += a_val * signs[28];
                    C_row[29] += a_val * signs[29];
                    C_row[30] += a_val * signs[30];
                    C_row[31] += a_val * signs[31];
                }
            }
        }
    }
}
