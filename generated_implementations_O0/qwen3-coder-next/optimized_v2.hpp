#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {  // For each row in A
        // Process all K columns of output row i together
        for (size_t j = 0; j < K; ++j) {
            float sum = 0.0f;
            
            // Unroll the inner loop for better performance
            size_t p = 0;
            for (; p + 7 < K; p += 8) {
                // Process 8 elements at a time
                for (size_t k = 0; k < 8; ++k) {
                    float a_val = A[i * K + p + k];
                    
                    // Extract the specific bit at B[p+k][j]
                    uint32_t packed = B[(p + k) * K_ints + (j / 32)];
                    uint32_t bit = (packed >> (j % 32)) & 1;
                    
                    float sign = bit ? 1.0f : -1.0f;
                    sum += a_val * sign;
                }
            }
            
            // Handle remaining elements
            for (; p < K; ++p) {
                float a_val = A[i * K + p];
                
                uint32_t packed = B[p * K_ints + (j / 32)];
                uint32_t bit = (packed >> (j % 32)) & 1;
                
                float sign = bit ? 1.0f : -1.0f;
                sum += a_val * sign;
            }
            
            C[i * K + j] = sum;
        }
    }
}
