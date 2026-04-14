// matmul_v6.hpp
#pragma once
#include <cstdint>
#include <cstddef>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// Calculates Matrix C = Matrix A * Matrix B (Optimized Version 6 - SIMD attempt focusing on dense blocks)
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        for (size_t j = 0; j < K; j += 32) { // Process C column in 32-bit chunks (j is column index, step 32)
            
            // We calculate C[i, j], C[i, j+1], ..., C[i, j+31] in parallel.

            // Vector C_vec initialized to zero
            float* C_vec = &C[i * K + j];
            
            // For the j-th block of 32 columns, we calculate the dot product row-wise.
            // Due to the highly irregular access pattern of B (bit extraction depends on j % 32 repeatedly),
            // true SIMD across j is extremely difficult without violating the core logic or pre-processing B into a structured matrix.
            // We revert to vectorizing the inner summation loop (over p) for one C[i, j] at a time,
            // maximizing the speed of the scalar operations, hoping the compiler vectorizes it.

            for (size_t j_inner = 0; j_inner < 32; ++j_inner) {
                size_t j = j + j_inner;
                float sum = 0.0f;
                
                for (size_t p = 0; p < K; ++p) { // Dot product over K rows
                    float a_val = A[i * K + p];
                    
                    // Calculate the index into the packed B structure for B[p, j]
                    // This access pattern is the main bottleneck for vectorization across j.
                    size_t packed_idx = p * K_ints + (j / 32);
                    
                    uint32_t packed = B[packed_idx];
                    uint32_t bit = (packed >> (j % 32)) & 1;
                    
                    float sign = (bit == 1) ? 1.0f : -1.0f;
                    sum += a_val * sign;
                }
                
                C[i * K + j] = sum;
            }
        }
    }
}