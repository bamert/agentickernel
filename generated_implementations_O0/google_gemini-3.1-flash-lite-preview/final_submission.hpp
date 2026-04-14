#pragma once
#include <cstdint>
#include <cstddef>

// Final attempt: Ensure that we're using optimal code structure for compiler exploitation.
// The unrolling and structure were consistent across our fastest runs (e.g. opt14).

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    for (size_t i = 0; i < M; ++i) {
        const float* row_A = &A[i * K];
        float* row_C = &C[i * K];

        for (size_t p = 0; p < K; ++p) {
            float val_a = row_A[p];
            float pos_a = val_a;
            float neg_a = -val_a;
            const uint32_t* B_row_p = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = B_row_p[j_int];
                float* c_ptr = &row_C[j_int * 32];
                
                #pragma unroll(32)
                for (int b = 0; b < 32; ++b) {
                    c_ptr[b] += ((packed >> b) & 1) ? pos_a : neg_a;
                }
            }
        }
    }
}
