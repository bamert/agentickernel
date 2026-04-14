#pragma once
#include <cstdint>
#include <cstddef>

// Final attempt: Refine v25 one last time.
// The code relies on the compiler unrolling the inner `col` loop.
// Adding __attribute__((always_inline)) or manually unrolling might help,
// but let's stick to the structure that was the most robust.

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* __restrict__ rowC = &C[i * K];
        const float* __restrict__ rowA = &A[i * K];

        // Zero-initialization
        for (size_t j = 0; j < K; ++j) rowC[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            const float val = rowA[p];
            const float n_val = -val;
            const uint32_t* __restrict__ rowB = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                const uint32_t packed = rowB[j_int];
                
                // Manual unroll of the `col` loop
                #define PROCESS_COL(col) \
                    do { \
                        const uint32_t bits = (packed >> ((col) * 4)) & 0xF; \
                        float res[4]; \
                        res[0] = (bits & 1) ? val : n_val; \
                        res[1] = (bits & 2) ? val : n_val; \
                        res[2] = (bits & 4) ? val : n_val; \
                        res[3] = (bits & 8) ? val : n_val; \
                        float* target = &rowC[j_int * 32 + (col) * 4]; \
                        vst1q_f32(target, vaddq_f32(vld1q_f32(target), vld1q_f32(res))); \
                    } while (0)

                PROCESS_COL(0);
                PROCESS_COL(1);
                PROCESS_COL(2);
                PROCESS_COL(3);
                PROCESS_COL(4);
                PROCESS_COL(5);
                PROCESS_COL(6);
                PROCESS_COL(7);
            }
        }
    }
}
