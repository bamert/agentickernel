#pragma once
#include <cstdint>
#include <cstddef>

// Optimization 9: Refine Opt6 to be as efficient as possible.
// Opt6 reached 344.19ms, significantly faster than baseline.
// We'll further optimize the inner loops and variable usage to see if 
// we can squeeze a little more performance out of it.
// Specifically, hoisting constants out and minimizing operations in the inner loop.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Reset C
    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    for (size_t i = 0; i < M; ++i) {
        const float* row_A = &A[i * K];
        float* row_C = &C[i * K];

        for (size_t p = 0; p < K; ++p) {
            float pos_a = row_A[p];
            float neg_a = -pos_a;
            
            const uint32_t* B_row_p = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = B_row_p[j_int];
                float* c_ptr = &row_C[j_int * 32];
                
                // Unrolling loop manually
                #pragma unroll
                for (int b = 0; b < 32; ++b) {
                    c_ptr[b] += ((packed >> b) & 1) ? pos_a : neg_a;
                }
            }
        }
    }
}
