#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>

// Optimized matmul with pointer arithmetic
// C = A * B where B is a K×K binary matrix packed as 1 bit = ±1.0f

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Initialize C to zero
    memset(C, 0, M * K * sizeof(float));
    
    for (size_t i = 0; i < M; ++i) {
        const float* A_row = A + i * K;
        float* C_row = C + i * K;
        
        for (size_t p = 0; p < K; ++p) {
            float a_val = A_row[p];
            const uint32_t* B_row = B + p * K_ints;
            float* C_row_ptr = C_row;
            
            // Process each 32-bit block
            for (size_t block = 0; block < K_ints; ++block) {
                uint32_t packed = B_row[block];
                
                // Compute all 32 signs quickly
                for (int k = 0; k < 32; ++k) {
                    *C_row_ptr++ += a_val * ((packed >> k) & 1 ? 1.0f : -1.0f);
                }
            }
        }
    }
}
