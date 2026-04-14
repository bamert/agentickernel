#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Zero C
    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    // Cache-friendly tiling/ordering
    // Process A in blocks of rows if M is large, but M=32 is small enough to fit.
    for (size_t i = 0; i < M; ++i) {
        float* Ci = &C[i * K];
        const float* Ai = &A[i * K];
        
        for (size_t p = 0; p < K; ++p) {
            const float a_val = Ai[p];
            const uint32_t* Bp = &B[p * K_ints];
            
            for (size_t kj = 0; kj < K_ints; ++kj) {
                const uint32_t bits = Bp[kj];
                float* Cij = &Ci[kj * 32];
                
                // Unrolling to reduce branch overhead and facilitate SIMD auto-vectorization
                // Manual unroll of 32 bits
                #define ACCUM(bit_idx) \
                    Cij[bit_idx] += ((bits >> (bit_idx)) & 1) ? a_val : -a_val;

                ACCUM(0)  ACCUM(1)  ACCUM(2)  ACCUM(3)
                ACCUM(4)  ACCUM(5)  ACCUM(6)  ACCUM(7)
                ACCUM(8)  ACCUM(9)  ACCUM(10) ACCUM(11)
                ACCUM(12) ACCUM(13) ACCUM(14) ACCUM(15)
                ACCUM(16) ACCUM(17) ACCUM(18) ACCUM(19)
                ACCUM(20) ACCUM(21) ACCUM(22) ACCUM(23)
                ACCUM(24) ACCUM(25) ACCUM(26) ACCUM(27)
                ACCUM(28) ACCUM(29) ACCUM(30) ACCUM(31)
                #undef ACCUM
            }
        }
    }
}
