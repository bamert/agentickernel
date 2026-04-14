#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    // v11 achieved ~14.7ms with 4-row tiling. 
    // Experimenting with a 2-row tiling to see if larger row counts 
    // are causing register pressure that hurts the compiler's auto-vectorization
    // of the inner 32-bit loop.
    for (size_t i = 0; i < M; i += 2) {
        float* Ci0 = &C[(i + 0) * K];
        float* Ci1 = &C[(i + 1) * K];

        const float* Ai0 = &A[(i + 0) * K];
        const float* Ai1 = &A[(i + 1) * K];

        for (size_t p = 0; p < K; ++p) {
            const float a0 = Ai0[p];
            const float a1 = Ai1[p];
            const uint32_t* Bp = &B[p * K_ints];
            
            for (size_t kj = 0; kj < K_ints; ++kj) {
                const uint32_t bits = Bp[kj];
                float* C0 = &Ci0[kj * 32];
                float* C1 = &Ci1[kj * 32];

                for (int b = 0; b < 32; ++b) {
                    float s = (bits & (1u << b)) ? 1.0f : -1.0f;
                    C0[b] += a0 * s;
                    C1[b] += a1 * s;
                }
            }
        }
    }
}
