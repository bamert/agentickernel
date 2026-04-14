#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Zero entire output matrix C
    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    // Process 8 rows at a time to maximize register reuse of packed B bits
    // M=32, so 8 rows divide evenly.
    for (size_t i = 0; i < M; i += 8) {
        for (size_t p = 0; p < K; ++p) {
            const float a0 = A[(i + 0) * K + p];
            const float a1 = A[(i + 1) * K + p];
            const float a2 = A[(i + 2) * K + p];
            const float a3 = A[(i + 3) * K + p];
            const float a4 = A[(i + 4) * K + p];
            const float a5 = A[(i + 5) * K + p];
            const float a6 = A[(i + 6) * K + p];
            const float a7 = A[(i + 7) * K + p];
            
            const uint32_t* Bp = &B[p * K_ints];
            
            for (size_t kj = 0; kj < K_ints; ++kj) {
                const uint32_t bits = Bp[kj];
                float* base = &C[kj * 32];
                float* Cij0 = &C[(i + 0) * K + kj * 32];
                float* Cij1 = &C[(i + 1) * K + kj * 32];
                float* Cij2 = &C[(i + 2) * K + kj * 32];
                float* Cij3 = &C[(i + 3) * K + kj * 32];
                float* Cij4 = &C[(i + 4) * K + kj * 32];
                float* Cij5 = &C[(i + 5) * K + kj * 32];
                float* Cij6 = &C[(i + 6) * K + kj * 32];
                float* Cij7 = &C[(i + 7) * K + kj * 32];

                for (int b = 0; b < 32; ++b) {
                    float s = (bits & (1u << b)) ? 1.0f : -1.0f;
                    Cij0[b] += a0 * s;
                    Cij1[b] += a1 * s;
                    Cij2[b] += a2 * s;
                    Cij3[b] += a3 * s;
                    Cij4[b] += a4 * s;
                    Cij5[b] += a5 * s;
                    Cij6[b] += a6 * s;
                    Cij7[b] += a7 * s;
                }
            }
        }
    }
}
