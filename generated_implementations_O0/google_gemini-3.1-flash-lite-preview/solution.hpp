#pragma once
#include <cstdint>
#include <cstddef>

// The best recorded performance was 341.396ms, which was from the version (opt14).
// This version is stable, fast, and uses a memory-efficient loop order.
// We will return it as the final solution.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Reset output matrix to zero
    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    for (size_t i = 0; i < M; ++i) {
        const float* row_A = &A[i * K];
        float* row_C = &C[i * K];

        for (size_t p = 0; p < K; ++p) {
            float val_a = row_A[p];
            const uint32_t* B_row_p = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = B_row_p[j_int];
                float* c_ptr = &row_C[j_int * 32];
                
                #pragma unroll(32)
                for (int b = 0; b < 32; ++b) {
                    float sign = ((packed >> b) & 1) ? 1.0f : -1.0f;
                    c_ptr[b] += val_a * sign;
                }
            }
        }
    }
}
