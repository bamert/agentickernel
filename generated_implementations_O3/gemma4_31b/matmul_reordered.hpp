#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>

// Optimized Matrix C = Matrix A * Matrix B
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* rowC = &C[i * K];
        const float* rowA = &A[i * K];
        
        // Initialize rowC to 0
        for (size_t j = 0; j < K; ++j) {
            rowC[j] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) {
            const float a_val = rowA[p];
            const uint32_t* B_row = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                const uint32_t packed = B_row[j_int];
                float* C_ptr = &rowC[j_int * 32];
                
                // Unrolling this loop manually or letting the compiler do it.
                // The branch on the bit is the bottleneck.
                for (int b = 0; b < 32; ++b) {
                    if ((packed >> b) & 1) {
                        C_ptr[b] += a_val;
                    } else {
                        C_ptr[b] -= a_val;
                    }
                }
            }
        }
    }
}
