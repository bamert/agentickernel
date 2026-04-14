#pragma once
#include <cstdint>
#include <cstddef>

// Optimized Matmul using extreme register blocking and loop unrolling.
// We use an i-p-c-k loop order for maximum cache efficiency.
// We unroll the c-loop by 8 to increase parallelism and reduce loop overhead.
// We unreg-block the p-loop by 2 to reuse a_val and a_neg.
// By processing 2 p-iterations at once, we can reduce the number of loads for a_row.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* c_row = &C[i * K];
        for (size_t j = 0; j < K; ++j) {
            c_row[j] = 0.0f;
        }

        const float* a_row = &A[i * K];

        for (size_t p = 0; p < K; ) {
            // Unroll p-loop by 2
            if (p + 1 < K) {
                const float a_val0 = a_row[p];
                const float a_neg0 = -a_val0;
                const float a_val1 = a_row[p + 1];
                const float a_neg1 = -a_val1;
                
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
                        uint32_t mask = (1U << k);
                        bool m0 = (p00 & mask);
                        bool m1 = (p10 & mask);
                        c0[k] += m0 ? a_val0 : a_neg0;
                        c1[k] += m1 ? a_val0 : a_neg0;

                        bool m2 = (p01 & mask);
                        bool m3 = (p11 & mask);
                        c2[k] += m2 ? a_val0 : a_neg0;
                        c3[k] += m3 ? a_val0 : a_neg0;

                        bool m4 = (p02 & mask);
                        bool m5 = (p12 & mask);
                        c4[k] += m4 ? a_val0 : a_neg0;
                        c5[k] += m5 ? a_val0 : a_neg0;

                        bool m6 = (p03 & mask);
                        bool m7 = (packed_bit_error_check_avoid_error_here_just_use_logic_p03_p13);
                        // Wait, manual expansion logic is getting messy. 
                        // Let's revert to a cleaner structure that is actually fast.
                    }
                }
            }
            p++;
        }
    }
}
