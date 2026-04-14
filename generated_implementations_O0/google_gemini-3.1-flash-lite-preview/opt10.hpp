#pragma once
#include <cstdint>
#include <cstddef>

// Retrying the loop order from opt6, but being careful about how the compiler handles 
// the ternary operator/bit check. 
// The compiler might generate a branch if we use a ternary. Let's force it to 
// be branchless if possible using standard arithmetic (sign = 2*bit - 1).

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
                
                // Using arithmetic to eliminate branching.
                // 1.0 if bit==1, -1.0 if bit==0
                // bit is (packed >> b) & 1
                // sign = 2.0f * (float)((packed >> b) & 1) - 1.0f;
                // Add val_a * sign = val_a * (2 * bit - 1) = val_a * 2 * bit - val_a
                
                for (int b = 0; b < 32; ++b) {
                    float bit = (float)((packed >> b) & 1);
                    c_ptr[b] += val_a * (2.0f * bit - 1.0f);
                }
            }
        }
    }
}
