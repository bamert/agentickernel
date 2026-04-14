#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Zero C
    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    // Unrolling more rows to improve throughput
    // A (MxK), B (KxK), C (MxK)
    // M=32, so we can unroll across rows of A
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
                float* Cij0 = &Ci0[kj * 32];
                float* Cij1 = &Ci1[kj * 32];
                float* Cij2 = &Ci2[kj * 32];
                float* Cij3 = &Ci3[kj * 32];

                #define STEP(b) \
                    { \
                        float sign = (bits & (1u << (b))) ? 1.0f : -1.0f; \
                        Cij0[b] += a0 * sign; \
                        Cij1[b] += a1 * sign; \
                        Cij2[b] += a2 * sign; \
                        Cij3[b] += a3 * sign; \
                    }

                STEP(0)  STEP(1)  STEP(2)  STEP(3)
                STEP(4)  STEP(5)  STEP(6)  STEP(7)
                STEP(8)  STEP(9)  STEP(10) STEP(11)
                STEP(12) STEP(13) STEP(14) STEP(15)
                STEP(16) STEP(17) STEP(18) STEP(19)
                STEP(20) STEP(21) STEP(22) STEP(23)
                STEP(24) STEP(25) STEP(26) STEP(27)
                STEP(28) STEP(29) STEP(30) STEP(31)
                #undef STEP
            }
        }
    }
}
