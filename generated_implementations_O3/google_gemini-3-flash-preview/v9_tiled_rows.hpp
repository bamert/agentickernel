#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Zero C
    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    // Use blocking/tiling for rows of A to keep Ci in L1
    // M=32 fits in cache, but let's process 4 rows at once to keep Bp reads recycled.
    for (size_t i = 0; i < M; i += 4) {
        for (size_t p = 0; p < K; ++p) {
            const float a0 = A[(i + 0) * K + p];
            const float a1 = A[(i + 1) * K + p];
            const float a2 = A[(i + 2) * K + p];
            const float a3 = A[(i + 3) * K + p];
            
            const uint32_t* Bp = &B[p * K_ints];
            
            for (size_t kj = 0; kj < K_ints; ++kj) {
                const uint32_t bits = Bp[kj];
                float* Cij0 = &C[(i + 0) * K + kj * 32];
                float* Cij1 = &C[(i + 1) * K + kj * 32];
                float* Cij2 = &C[(i + 2) * K + kj * 32];
                float* Cij3 = &C[(i + 3) * K + kj * 32];

                for (int b = 0; b < 32; ++b) {
                    float s = (bits & (1u << b)) ? 1.0f : -1.0f;
                    Cij0[b] += a0 * s;
                    Cij1[b] += a1 * s;
                    Cij2[b] += a2 * s;
                    Cij3[b] += a3 * s;
                }
            }
        }
    }
}
