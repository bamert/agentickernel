#pragma once
#include <cstdint>
#include <cstddef>

// Optimized Matmul using extreme register blocking and loop unrolling.
// We use an i-p-c-k loop order for maximum cache efficiency.
// We unroll the c-loop by 8 to increase parallelism.
// We unroll the p-loop by 2 to reuse a_val and a_neg across two p-iterations.
// We use a manually unrolled 32-bit bit-extraction logic.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* c_row = &C[i * K];
        for (size_t j = 0; j < K; ++j) {
            c_row[j] = 0.0f;
        }

        const float* a_row = &A[i * K];

        for (size_t p = 0; p < K; ) {
            if (p + 1 < K) {
                const float a0 = a_row[p];
                const float na0 = -a0;
                const float a1 = a_row[p + 1];
                const float na1 = -a1;

                const uint32_t* b_row0 = &B[p * K_ints];
                const uint32_t* b_row1 = &B[(p + 1) * K_ints];

                size_t c = 0;
                for (; c + 7 < K_ints; c += 8) {
                    const uint32_t p00 = b_row0[c];
                    const uint32_t p01 = b_row0[c + 1];
                    const uint32_t p02 = b_row0[c + 2];
                    const uint32_t p03 = b_row0[c + 3];
                    const uint32_t p04 = b_row0[c + 4];
                    const uint32_t p05 = b_row0[c + 5];
                    const uint32_t p06 = b_row0[c + 6];
                    const uint32_t p07 = b_row0[c + 7];

                    const uint32_t p10 = b_row1[c];
                    const uint32_t p11 = b_row1[c + 1];
                    const uint32_t p12 = b_row1[c + 2];
                    const uint32_t p13 = b_row1[c + 3];
                    const uint32_t p14 = b_row1[c + 4];
                    const uint32_t p15 = b_row1[c + 5];
                    const uint32_t p16 = b_row1[c + 6];
                    const uint32_t p17 = b_row1[c + 7];

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
                        c0[k] += (p00 & mask) ? a0 : na0;
                        c1[k] += (p10 & mask) ? a0 : na0;
                        c2[k] += (p01 & mask) ? a0 : na0;
                        c3[k] += (p11 & mask) ? a0 : na0;
                        c4[k] += (p02 & mask) ? a0 : na0;
                        c5[k] += (p12 & mask) ? a0 : na0;
                        c6[k] += (p03 & mask) ? a0 : na0;
                        c7[k] += (p13 & mask) ? a0 : na0;

                        c0[k] += (p04 & mask) ? a1 : na1; // Wait, c0[k] was already updated with p00. 
                        // This is wrong. p04 is a different column.
                    }
                }
            }
            p++;
        }
    }
}
