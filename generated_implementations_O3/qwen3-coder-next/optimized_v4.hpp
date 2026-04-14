#pragma once
#include <cstdint>
#include <cstddef>

// Optimized matmul with better bit manipulation and memory access
// C = A * B where B is a K×K binary matrix packed as 1 bit = ±1.0f

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Initialize C to zero
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }
    }
    
    // i-p-j loop order
    for (size_t i = 0; i < M; ++i) {
        const float* A_row = A + i * K;
        float* C_row = C + i * K;
        
        for (size_t p = 0; p < K; ++p) {
            float a_val = A_row[p];
            const uint32_t* B_row = B + p * K_ints;
            
            // Precompute sign values for this row p of B
            // Store in a temporary array to avoid repeated bit manipulation
            float signs[32];
            for (size_t b = 0; b < 32; ++b) {
                // Generate all possible sign combinations for a 32-bit group
                uint32_t mask = 1u << b;
                signs[b] = (B_row[0] & mask) ? 1.0f : -1.0f;
            }
            
            // Process first 32 columns using precomputed signs
            for (size_t j = 0; j < 32; ++j) {
                C_row[j] += a_val * signs[j];
            }
            
            // Process remaining 32-bit blocks
            for (size_t block = 1; block < K_ints; ++block) {
                uint32_t packed = B_row[block];
                float block_signs[32];
                
                // Precompute signs for this block
                for (size_t b = 0; b < 32; ++b) {
                    block_signs[b] = (packed & (1u << b)) ? 1.0f : -1.0f;
                }
                
                size_t base_j = block * 32;
                for (size_t j = 0; j < 32; ++j) {
                    C_row[base_j + j] += a_val * block_signs[j];
                }
            }
        }
    }
}
