#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Precompute signs with byte-level lookup table
    alignas(32) float luts[256][8];  // Lookup table for 8-bit bytes
    
    // Initialize lookup table: byte -> 8 signs
    for (int b = 0; b < 256; ++b) {
        for (int bit = 0; bit < 8; ++bit) {
            luts[b][bit] = ((b >> bit) & 1) ? 1.0f : -1.0f;
        }
    }

    // Process with i, j, p order but compute signs on-the-fly
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            float sum = 0.0f;
            
            // Process p in chunks of 32, but use lookup tables
            size_t p = 0;
            for (; p + 31 < K; p += 32) {
                // Process 32 consecutive p values at once
                for (int k = 0; k < 32; ++k) {
                    float a_val = A[i * K + p + k];
                    const uint32_t* b_row = B + (p + k) * K_ints;
                    
                    // Get the word containing column j
                    uint32_t word = b_row[j / 32];
                    // Extract the bit
                    uint32_t bit = (word >> (j % 32)) & 1;
                    
                    float sign = bit ? 1.0f : -1.0f;
                    sum += a_val * sign;
                }
            }
            
            // Handle remaining p values
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
