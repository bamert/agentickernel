#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    // Process single row at a time.
    // Focusing strictly on the loop structure that allowed v12 and v11 to be fast.
    for (size_t i = 0; i < M; ++i) {
        float* Ci = &C[i * K];
        const float* Ai = &A[i * K];

        for (size_t p = 0; p < K; ++p) {
            const float a = Ai[p];
            const uint32_t* Bp = &B[p * K_ints];
            
            for (size_t kj = 0; kj < K_ints; ++kj) {
                const uint32_t bits = Bp[kj];
                float* Cij = &Ci[kj * 32];

                // Inner loop unrolling explicitly to help with SIMD
                for (int b = 0; b < 32; ++b) {
                    float s = (bits & (1u << b)) ? a : -a;
                    Cij[b] += s;
                }
            }
        }
    }
}
