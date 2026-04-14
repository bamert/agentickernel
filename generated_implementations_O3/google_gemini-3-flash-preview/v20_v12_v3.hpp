#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    // v12 style 2-row tiling is the sweet spot for the compiler.
    // Let's ensure the compiler treats the inner loop as bit manipulation.
    for (size_t i = 0; i < M; i += 2) {
        float* Ci0 = &C[i * K];
        float* Ci1 = &C[(i + 1) * K];
        const float* Ai0 = &A[i * K];
        const float* Ai1 = &A[(i + 1) * K];

        for (size_t p = 0; p < K; ++p) {
            const float a0 = Ai0[p];
            const float a1 = Ai1[p];
            const uint32_t* Bp = &B[p * K_ints];
            
            for (size_t kj = 0; kj < K_ints; ++kj) {
                const uint32_t bits = Bp[kj];
                float* C0 = &Ci0[kj * 32];
                float* C1 = &Ci1[kj * 32];

                // Hinting the compiler into a more efficient form.
                for (int b = 0; b < 32; ++b) {
                    // Extract bit as 0 or 1
                    uint32_t bit = (bits >> b) & 1;
                    // Compute sign: 1.0f if bit is 1, -1.0f if bit is 0
                    float s = (float)(int(bit) * 2 - 1);
                    C0[b] += a0 * s;
                    C1[b] += a1 * s;
                }
            }
        }
    }
}
