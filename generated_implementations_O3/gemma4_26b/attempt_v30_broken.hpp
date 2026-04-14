#pragma once
#include <cstdint>
#include <cstddef>

// Optimized Matmul using effective register blocking.
// We unroll the p-loop (rows of B) and c-loop (columns of B).
// We avoid the mistake of updating the same C element with different p-values 
// from different rows in a way that overwrites the work of the previous p-iteration.
// Instead, we use the i-p-c-k order and process p-iterations sequentially.
// To optimize, we unroll the c-loop (columns) to increase instruction level parallelism.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* c_row = &C[i * K];
        for (size_t j = 0; j < K; ++j) {
            c_row[j] = 0.0f;
        }

        const float* a_row = &A[i * K];

        for (size_t p = 0; p < K; ++p) {
            const float a_val = a_row[p];
            const float a_neg = -a_val;
            const uint32_t* b_row = &B[p * K_ints];

            size_t c = 0;
            // Unroll c-loop by 8
            for (; c + 7 < K_ints; c += 8) {
                const uint32_t p0 = b_row[c];
                const uint32_t p1 = b_row[c + 1];
                const uint32_t p2 = b_row[c + 2];
                const uint32_t p3 = b_row[c + 3];
                const uint32_t p4 = b_row[c + 4];
                const uint32_t p5 = b_row[c + 5];
                const uint32_t p6 = b_row[c + 6];
                const uint32_t p7 = b_row[c + 7];

                float* c0 = &c_row[c * 32];
                float* c1 = &c_row[(c + 1) * 32];
                float* c2 = &c_row[(c + 2) * 32];
                float* c3 = &c_row[(c + 3) * 32];
                float* c4 = &c_row[(c + 4) * 32];
                float* c5 = &c_row[(c + 5) * 32];
                float* c6 = &c_row[(c + 6) * 32];
                float* c7 = &c_row[(c + 7) * 32];

                for (size_t k = 0; k < 32; ++k) {
                    const uint32_t mask = (1U << k);
                    c0[k] += (p0 & mask) ? a_val : a_neg;
                    c1[k] += (p1 & mask) ? a_val : a_neg;
                    c2[k] += (p2 & mask) ? a_val : a_neg;
                    c3[k] += (par_extract(p3, k, a_val, a_neg)); // Wait, no.
                    // Just do it clearly.
                }
            }
        }
    }
}
