#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    // v9 was 4-row tiled. Let's stick to 4 rows but optimize bit access.
    for (size_t i = 0; i < M; i += 4) {
        float* Ci0 = &C[(i + 0) * K];
        float* Ci1 = &C[(i + 1) * K];
        float* Ci2 = &C[(i + 2) * K];
        float* Ci3 = &C[(i + 3) * K];

        const float* Ai0 = &A[(i + 0) * K];
        const float* Ai1 = &A[(i + 1) * K];
        const float* Ai2 = &A[(i + 2) * K];
        const float* Ai3 = &A[(i + 3) * K];

        for (size_t p = 0; p < K; ++p) {
            const float a0 = Ai0[p];
            const float a1 = Ai1[p];
            const float a2 = Ai2[p];
            const float a3 = Ai3[p];
            
            const uint32_t* Bp = &B[p * K_ints];
            
            for (size_t kj = 0; kj < K_ints; ++kj) {
                const uint32_t bits = Bp[kj];
                float* C0 = &Ci0[kj * 32];
                float* C1 = &Ci1[kj * 32];
                float* C2 = &Ci2[kj * 32];
                float* C3 = &Ci3[kj * 32];

                // Manually unroll the bit-loop to help compiler avoid indexing logic
                for (int b = 0; b < 32; ++b) {
                    float mask = (bits & (1u << b)) ? 1.0f : -1.0f;
                    C0[b] += a0 * mask;
                    C1[b] += a1 * mask;
                    C2[b] += a2 * mask;
                    C3[b] += a3 * mask;
                }
            }
        }
    }
}
