#pragma once
#include <cstdint>
#include <cstddef>

// Optimized matmul using branchless sign extraction and loop unrolling
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K >> 5;  // K / 32, using shift for speed

    // pragma unroll to reduce loop overhead
    #pragma unroll 4
    for (size_t i = 0; i < M; ++i) {         // Each row of A
        for (size_t j = 0; j < K; ++j) {     // Each column of B
            float sum = 0.0f;
            
            // Compute word index and mask for column j in constant time
            size_t word_idx = j >> 5;           // j / 32
            uint32_t bit_pos = (uint32_t)(j & 31); // bit position within word
            
            // Extract the bit and convert to sign (+1 or -1) without branching
            // bit = ((B[p*K_ints + word_idx] >> bit_pos) & 1)
            // sign = (bit << 1) - 1  =>  -1 if bit==0, +1 if bit==1
            for (size_t p = 0; p < K; ++p) {   // Dot product across rows
                float a_val = A[i * K + p];
                
                // Load packed word for row p
                uint32_t packed = B[p * K_ints + word_idx];
                
                // Get the specific bit and turn it into +1.0f or -1.0f
                // ( (packed >> bit_pos) & 1 ) << 1  yields 0 or 2
                // Subtract 1 to get -1 or +1
                float sign = ( ((packed >> bit_pos) & 1u) << 1 ) - 1.0f;
                
                sum += a_val * sign;
            }
            
            C[i * K + j] = sum;
        }
    }
}