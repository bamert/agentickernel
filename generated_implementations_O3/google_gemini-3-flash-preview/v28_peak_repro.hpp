#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    // Use 2-row tiling (M=32)
    // The constant pointers and simplified inner loop help the compiler vectorize effectively.
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

                // Attempt to reach the 14.16ms peak performance by keeping the loop as tight as possible.
                // The compiler's ability to transform this into efficient SIMD instructions (FMLA/FMSB)
                // is key to achieving high throughput.
                for (int b = 0; b < 32; ++b) {
                    float s = (bits & (1u << b)) ? a0 : -a0;
                    C0[b] += s;
                    float s2 = (bits & (1u << b)) ? a1 : -a1;
                    C1[b] += s2;
                }
            }
        }
    }
}
