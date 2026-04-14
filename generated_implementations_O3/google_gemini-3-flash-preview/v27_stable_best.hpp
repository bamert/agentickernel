#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    // Use 2-row tiling (M=32)
    for (size_t i = 0; i < M; i += 2) {
        float* const Ci0 = &C[i * K];
        float* const Ci1 = &C[(i + 1) * K];
        const float* const Ai0 = &A[i * K];
        const float* const Ai1 = &A[(i + 1) * K];

        for (size_t p = 0; p < K; ++p) {
            const float a0 = Ai0[p];
            const float a1 = Ai1[p];
            const uint32_t* const Bp = &B[p * K_ints];
            
            for (size_t kj = 0; kj < K_ints; ++kj) {
                const uint32_t bits = Bp[kj];
                float* const C0 = &Ci0[kj * 32];
                float* const C1 = &Ci1[kj * 32];

                for (int b = 0; b < 32; ++b) {
                    // Using a bitmask-friendly way to avoid any branching
                    // and maximize auto-vectorization potential.
                    float sign = (bits & (1u << b)) ? 1.0f : -1.0f;
                    C0[b] += a0 * sign;
                    C1[b] += a1 * sign;
                }
            }
        }
    }
}
