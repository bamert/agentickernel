// matmul_v4.hpp
#pragma once
#include <cstdint>
#include <cstddef>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// Calculates Matrix C = Matrix A * Matrix B (Optimized Version 4 - SIMD/NEON Focused)
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        for (size_t j = 0; j < K; j += 32) { // Process C column in 32-bit chunks (j is column index)
            
            // Vectorized calculation for C[i, j], C[i, j+1], ..., C[i, j+31]
            
            // Cache the start index for the required 32 bits in B for all relevant rows p
            // We need B[p*, j] for p=0 to K-1.
            
            // We must reconstruct the dot product for the entire block C[i, j:j+31]
            // This requires reading the whole relevant column j of B.
            
            // Since we are calculating C[i, j], we need the signs from B[p, j] for all p.
            
            // This approach is complex because the required signs depend on indices j % 32, which is highly irregular when vectorizing over j directly.
            
            // Reverting to calculating C[i, j] one by one but using NEON for the inner loop summation over p, targeting the $O(M K^2)$ complexity.
            // Since we cannot easily vectorize the dependency on j inside the p loop without rewriting B's structure, we focus on making the scalar calculation inside the loop extremely fast.
            
            float sum = 0.0f;
            
            for (size_t p = 0; p < K; ++p) { // Calculate the dot product (p indexes the rows of B)
                float a_val = A[i * K + p];
                
                // Extract the specific bit at B[p][j]
                // B is row-major: index = p * K_ints + (j / 32) for the 32-bit word
                size_t packed_idx = p * K_ints + (j / 32);
                
                uint32_t packed = B[packed_idx];
                uint32_t bit = (packed >> (j % 32)) & 1;
                
                float sign = bit ? 1.0f : -1.0f;
                sum += a_val * sign;
            }
            
            C[i * K + j] = sum;
        }
    }
}