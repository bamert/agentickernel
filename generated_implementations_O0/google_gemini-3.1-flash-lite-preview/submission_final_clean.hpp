#pragma once
#include <cstdint>
#include <cstddef>

/**
 * MatMul (i, p, K_ints, b) loop order.
 * This structure achieved ~341ms in several tests.
 * 
 * Final cleanup: ensuring pointers are const-correct and variables
 * are minimal.
 */
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    // Reset output matrix C to zero
    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    for (size_t i = 0; i < M; ++i) {
        const float* row_A  = &A[i * K];
        float*       row_C  = &C[i * K];

        for (size_t p = 0; p < K; ++p) {
            const float val_a = row_A[p];
            const float pos_a =  val_a;
            const float neg_a = -val_a;

            const uint32_t* B_row_p = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                const uint32_t packed = B_row_p[j_int];
                float* c_ptr = &row_C[j_int * 32];
                
                #pragma unroll(32)
                for (int b = 0; b < 32; ++b) {
                    c_ptr[b] += ((packed >> b) & 1) ? pos_a : neg_a;
                }
            }
        }
    }
}
