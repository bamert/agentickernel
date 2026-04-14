#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    // Use 2-row tiling as it was fastest.
    // Further unroll the inner bit loop to reduce overhead.
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

                #define BSTEP(b) { \
                    float s = (bits & (1u << (b))) ? 1.0f : -1.0f; \
                    C0[b] += a0 * s; \
                    C1[b] += a1 * s; \
                }

                BSTEP(0)  BSTEP(1)  BSTEP(2)  BSTEP(3)
                BSTEP(4)  BSTEP(5)  BSTEP(6)  BSTEP(7)
                BSTEP(8)  BSTEP(9)  BSTEP(10) BSTEP(11)
                BSTEP(12) BSTEP(13) BSTEP(14) BSTEP(15)
                BSTEP(16) BSTEP(17) BSTEP(18) BSTEP(19)
                BSTEP(20) BSTEP(21) BSTEP(22) BSTEP(23)
                BSTEP(24) BSTEP(25) BSTEP(26) BSTEP(27)
                BSTEP(28) BSTEP(29) BSTEP(30) BSTEP(31)
                #undef BSTEP
            }
        }
    }
}
