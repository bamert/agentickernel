#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Process in blocks to improve cache utilization
    const int BLOCK_K = 128;  // Block size for K dimension
    
    // Initialize output matrix to zero
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }
    }

    // i, p, j loop ordering with blocking
    for (size_t i = 0; i < M; ++i) {
        for (size_t p_block = 0; p_block < K; p_block += BLOCK_K) {
            size_t p_end = std::min(p_block + BLOCK_K, K);
            
            // Process this block of p values
            for (size_t p = p_block; p < p_end; ++p) {
                float a_val = A[i * K + p];
                
                // Accumulate contribution of this p to all j values
                // Process j in blocks for better cache utilization
                for (size_t j_block = 0; j_block < K; j_block += BLOCK_K) {
                    size_t j_end = std::min(j_block + BLOCK_K, K);
                    
                    // Convert bits to signs for this row segment
                    for (size_t j = j_block; j < j_end; ++j) {
                        // Extract bit from B[p][j]
                        uint32_t packed = B[p * K_ints + (j / 32)];
                        uint32_t bit = (packed >> (j % 32)) & 1;
                        float sign = bit ? 1.0f : -1.0f;
                        
                        C[i * K + j] += a_val * sign;
                    }
                }
            }
        }
    }
}
