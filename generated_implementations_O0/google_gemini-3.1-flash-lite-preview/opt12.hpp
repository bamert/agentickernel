#pragma once
#include <cstdint>
#include <cstddef>

// Optimization 12: Reverting to the logic of opt10 which was the fastest (344.05ms).
// It used bit extraction followed by arithmetic to be branchless.
// We will keep this structure, as it appears the compiler effectively optimizes it.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

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
                
                for (int b = 0; b < 32; ++b) {
                    float bit = (float)((packed >> b) & 1);
                    c_ptr[b] += val_a * (2.0f * bit - 1.0f);
                }
            }
        }
    }
}
