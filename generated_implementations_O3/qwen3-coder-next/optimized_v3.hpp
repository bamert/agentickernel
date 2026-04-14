#pragma once
#include <cstdint>
#include <cstddef>

// Further optimized matmul: C = A * B
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// K is guaranteed to be a multiple of 32.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Initialize C to zero first
    size_t total_C = M * K;
    for (size_t idx = 0; idx < total_C; ++idx) {
        C[idx] = 0.0f;
    }

    // Loop order i-p-j for better cache locality
    for (size_t i = 0; i < M; ++i) {
        const float* A_row = A + i * K;
        float* C_row = C + i * K;
        
        for (size_t p = 0; p < K; ++p) {
            float a_val = A_row[p];
            
            // Process 4 columns at a time to reduce loop overhead
            size_t j = 0;
            for (; j + 3 < K; j += 4) {
                // Calculate indices for 4 consecutive columns
                size_t int_idx0 = j / 32;
                size_t bit_pos0 = j % 32;
                size_t int_idx1 = (j+1) / 32;
                size_t bit_pos1 = (j+1) % 32;
                size_t int_idx2 = (j+2) / 32;
                size_t bit_pos2 = (j+2) % 32;
                size_t int_idx3 = (j+3) / 32;
                size_t bit_pos3 = (j+3) % 32;
                
                uint32_t packed = B[p * K_ints + int_idx0];
                uint32_t bit0 = (packed >> bit_pos0) & 1;
                uint32_t bit1 = (packed >> bit_pos1) & 1;
                uint32_t bit2 = (packed >> bit_pos2) & 1;
                uint32_t bit3 = (packed >> bit_pos3) & 1;
                
                float sign0 = bit0 ? 1.0f : -1.0f;
                float sign1 = bit1 ? 1.0f : -1.0f;
                float sign2 = bit2 ? 1.0f : -1.0f;
                float sign3 = bit3 ? 1.0f : -1.0f;
                
                C_row[j] += a_val * sign0;
                C_row[j+1] += a_val * sign1;
                C_row[j+2] += a_val * sign2;
                C_row[j+3] += a_val * sign3;
            }
            
            // Handle remaining columns
            for (; j < K; ++j) {
                uint32_t packed = B[p * K_ints + (j / 32)];
                uint32_t bit = (packed >> (j % 32)) & 1;
                float sign = bit ? 1.0f : -1.0f;
                
                C_row[j] += a_val * sign;
            }
        }
    }
}
