#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>

// Further optimized matmul - eliminate temporary sign arrays
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
            
            // Process each 32-bit block
            for (size_t block = 0; block < K_ints; ++block) {
                uint32_t packed = B_row[block];
                size_t base_j = block * 32;
                
                // Process 32 columns in this block, unrolled
                C_row[base_j + 0]  += a_val * ((packed >> 0)  & 1 ? 1.0f : -1.0f);
                C_row[base_j + 1]  += a_val * ((packed >> 1)  & 1 ? 1.0f : -1.0f);
                C_row[base_j + 2]  += a_val * ((packed >> 2)  & 1 ? 1.0f : -1.0f);
                C_row[base_j + 3]  += a_val * ((packed >> 3)  & 1 ? 1.0f : -1.0f);
                C_row[base_j + 4]  += a_val * ((packed >> 4)  & 1 ? 1.0f : -1.0f);
                C_row[base_j + 5]  += a_val * ((packed >> 5)  & 1 ? 1.0f : -1.0f);
                C_row[base_j + 6]  += a_val * ((packed >> 6)  & 1 ? 1.0f : -1.0f);
                C_row[base_j + 7]  += a_val * ((packed >> 7)  & 1 ? 1.0f : -1.0f);
                C_row[base_j + 8]  += a_val * ((packed >> 8)  & 1 ? 1.0f : -1.0f);
                C_row[base_j + 9]  += a_val * ((packed >> 9)  & 1 ? 1.0f : -1.0f);
                C_row[base_j + 10] += a_val * ((packed >> 10) & 1 ? 1.0f : -1.0f);
                C_row[base_j + 11] += a_val * ((packed >> 11) & 1 ? 1.0f : -1.0f);
                C_row[base_j + 12] += a_val * ((packed >> 12) & 1 ? 1.0f : -1.0f);
                C_row[base_j + 13] += a_val * ((packed >> 13) & 1 ? 1.0f : -1.0f);
                C_row[base_j + 14] += a_val * ((packed >> 14) & 1 ? 1.0f : -1.0f);
                C_row[base_j + 15] += a_val * ((packed >> 15) & 1 ? 1.0f : -1.0f);
                C_row[base_j + 16] += a_val * ((packed >> 16) & 1 ? 1.0f : -1.0f);
                C_row[base_j + 17] += a_val * ((packed >> 17) & 1 ? 1.0f : -1.0f);
                C_row[base_j + 18] += a_val * ((packed >> 18) & 1 ? 1.0f : -1.0f);
                C_row[base_j + 19] += a_val * ((packed >> 19) & 1 ? 1.0f : -1.0f);
                C_row[base_j + 20] += a_val * ((packed >> 20) & 1 ? 1.0f : -1.0f);
                C_row[base_j + 21] += a_val * ((packed >> 21) & 1 ? 1.0f : -1.0f);
                C_row[base_j + 22] += a_val * ((packed >> 22) & 1 ? 1.0f : -1.0f);
                C_row[base_j + 23] += a_val * ((packed >> 23) & 1 ? 1.0f : -1.0f);
                C_row[base_j + 24] += a_val * ((packed >> 24) & 1 ? 1.0f : -1.0f);
                C_row[base_j + 25] += a_val * ((packed >> 25) & 1 ? 1.0f : -1.0f);
                C_row[base_j + 26] += a_val * ((packed >> 26) & 1 ? 1.0f : -1.0f);
                C_row[base_j + 27] += a_val * ((packed >> 27) & 1 ? 1.0f : -1.0f);
                C_row[base_j + 28] += a_val * ((packed >> 28) & 1 ? 1.0f : -1.0f);
                C_row[base_j + 29] += a_val * ((packed >> 29) & 1 ? 1.0f : -1.0f);
                C_row[base_j + 30] += a_val * ((packed >> 30) & 1 ? 1.0f : -1.0f);
                C_row[base_j + 31] += a_val * ((packed >> 31) & 1 ? 1.0f : -1.0f);
            }
        }
    }
}
