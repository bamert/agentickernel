#pragma once
#include <cstdint>
#include <cstddef>

// Optimized matmul with block processing for better performance
// C = A * B where B is a K×K binary matrix packed as 1 bit = ±1.0f

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Initialize C to zero using memset for better performance
    memset(C, 0, M * K * sizeof(float));
    
    // Process in blocks to improve cache utilization
    // Block size should fit in L1 cache
    const int BLOCK_SIZE = 64;  // Process 64 rows of A at a time
    
    for (size_t i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
        size_t i1 = (i0 + BLOCK_SIZE <= M) ? i0 + BLOCK_SIZE : M;
        
        for (size_t p = 0; p < K; ++p) {
            const float* A_rows = A + i0 * K;
            const uint32_t* B_row = B + p * K_ints;
            float* C_rows = C + i0 * K;
            
            float a_val = A_rows[p];
            
            // Process B in 32-bit blocks
            for (size_t block = 0; block < K_ints; ++block) {
                uint32_t packed = B_row[block];
                
                // Use compiler's built-in popcount or bit manipulations
                float signs[32];
                for (int bit = 0; bit < 32; ++bit) {
                    signs[bit] = (packed & (1u << bit)) ? 1.0f : -1.0f;
                }
                
                size_t base_j = block * 32;
                for (size_t j = 0; j < 32; ++j) {
                    floatcontrib = a_val * signs[j];
                    for (size_t i = 0; i < i1 - i0; ++i) {
                        C_rows[i * K + base_j + j] += contrib;
                    }
                }
            }
        }
    }
}
