#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    // Exact structure of v12 but with potentially better pointer arithmetic
    for (size_t i = 0; i < M; i += 2) {
        for (size_t p = 0; p < K; ++p) {
            const float a0 = A[i * K + p];
            const float a1 = A[(i + 1) * K + p];
            const uint32_t* Bp = &B[p * K_ints];
            
            for (size_t kj = 0; kj < K_ints; ++kj) {
                const uint32_t bits = Bp[kj];
                float* C0 = &C[i * K + kj * 32];
                float* C1 = &C[(i + 1) * K + kj * 32];

                for (int b = 0; b < 32; ++b) {
                    float s = (bits & (1u << b)) ? 1.0f : -1.0f;
                    C0[b] += a0 * s;
                    C1[b] += a1 * s;
                }
            }
        }
    }
}
