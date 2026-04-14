#pragma once
#include <cstdint>
#include <cstddef>

// Optimization 11: Final tweak of opt10. 
// Instead of 2.0f * bit - 1.0f which involves float multiplication and addition,
// we can use a lookup table or conditional pointer selection to add either +val_a or -val_a.
// Given opt10 performs well, the key is the compiler's ability to vectorize the loop.
// Let's ensure the loop is as simple as possible to maximize potential autovectorization.

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
                
                // Keep it very simple to encourage the compiler to use SIMD (e.g., vbslQ_f32)
                for (int b = 0; b < 32; ++b) {
                    // This ternary is often compiled into a conditional selection (csel) on ARM
                    c_ptr[b] += ((packed >> b) & 1) ? pos_a : neg_a;
                }
            }
        }
    }
}
