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
                    const float* sign_p = signs;
                    
                    // Use pointer arithmetic in inner loop for optimal performance
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                    *C_row++ += a_val * *sign_p++;
                }
            }
        }
    }
}
