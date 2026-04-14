#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    // v12 style 2-row tiling remains the king.
    // Let's try to unroll the 'p' loop slightly to see if we can hide some latency.
    for (size_t i = 0; i < M; i += 2) {
        float* Ci0 = &C[(i + 0) * K];
        float* Ci1 = &C[(i + 1) * K];

        const float* Ai0 = &A[(i + 0) * K];
        const float* Ai1 = &A[(i + 1) * K];

        for (size_t p = 0; p < K; p += 2) {
            // P = 0
            const float a0_0 = Ai0[p];
            const float a1_0 = Ai1[p];
            const uint32_t* Bp0 = &B[p * K_ints];
            
            // P = 1
            const float a0_1 = Ai0[p+1];
            const float a1_1 = Ai1[p+1];
            const uint32_t* Bp1 = &B[(p+1) * K_ints];
            
            for (size_t kj = 0; kj < K_ints; ++kj) {
                const uint32_t bits0 = Bp0[kj];
                const uint32_t bits1 = Bp1[kj];
                float* C0 = &Ci0[kj * 32];
                float* C1 = &Ci1[kj * 32];

                for (int b = 0; b < 32; ++b) {
                    float s0 = (bits0 & (1u << b)) ? a0_0 : -a0_0;
                    float s1 = (bits1 & (1u << b)) ? a0_1 : -a0_1;
                    C0[b] += s0 + s1;
                    
                    float s2 = (bits0 & (1u << b)) ? a1_0 : -a1_0;
                    float s3 = (bits1 & (1u << b)) ? a1_1 : -a1_1;
                    C1[b] += s2 + s3;
                }
            }
        }
    }
}
