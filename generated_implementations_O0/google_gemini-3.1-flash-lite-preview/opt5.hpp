#pragma once
#include <cstdint>
#include <cstddef>

// Optimization 5: Refine the successful Opt1. 
// The primary gain in Opt1 came from loop reordering (M -> P -> K),
// which allows us to process contiguous memory in C and reduces pointer arithmetic/indexing.
// We keep the loop structure but optimize the inner-most loop for better performance.
// - Use pointer arithmetic.
// - Limit divisions/modulos.
// - Unroll the bit loop.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Reset C
    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    for (size_t i = 0; i < M; ++i) {
        const float* row_A = &A[i * K];
        float* row_C = &C[i * K];

        for (size_t p = 0; p < K; ++p) {
            float a_val = row_A[p];
            float pos_a = a_val;
            float neg_a = -a_val;
            
            const uint32_t* row_B_packed = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = row_B_packed[j_int];
                float* c_ptr = &row_C[j_int * 32];
                
                // Manual unroll of the bit loop
                for (int bit = 0; bit < 32; ++bit) {
                    c_ptr[bit] += ((packed >> bit) & 1) ? pos_a : neg_a;
                }
            }
        }
    }
}
