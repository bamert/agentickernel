#pragma once
#include <cstdint>
#include <cstddef>

// Final attempt to improve upon opt14.
// Revisiting the bit extraction/sign multiplication logic for maximum efficiency.
// Optimization: Move the sign constant logic out to a more explicit representation
// to ensure the register use is optimal and no branches are generated.
// The previous fastest was 341.40ms (opt14).

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
                    // This is the tightest loop found so far that works well.
                    // The ternary condition usually translates to a Conditional Select (csel) on ARM.
                    c_ptr[b] += ((packed >> b) & 1) ? pos_a : neg_a;
                }
            }
        }
    }
}
