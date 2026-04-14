#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>

// Optimized matmul with block processing for better performance
// C = A * B where B is a K×K binary matrix packed as 1 bit = ±1.0f

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Initialize C to zero using memset for better performance
    memset(C, 0, M * K * sizeof(float));
    
    // i-p-j loop order with block processing
    for (size_t i = 0; i < M; ++i) {
        const float* A_row = A + i * K;
        float* C_row = C + i * K;
        
        for (size_t p = 0; p < K; ++p) {
            float a_val = A_row[p];
            const uint32_t* B_row = B + p * K_ints;
            
            // Process in blocks of 32 columns
            for (size_t block = 0; block < K_ints; ++block) {
                uint32_t packed = B_row[block];
                
                // Precompute signs for this 32-bit block
                float signs[32];
                for (int bit = 0; bit < 32; ++bit) {
                    signs[bit] = (packed & (1u << bit)) ? 1.0f : -1.0f;
                }
                
                size_t base_j = block * 32;
                for (size_t j = 0; j < 32; ++j) {
                    C_row[base_j + j] += a_val * signs[j];
                }
            }
        }
    }
}
