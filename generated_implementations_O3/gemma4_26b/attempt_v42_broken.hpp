#pragma once
#include <cstdint>
#include <cstddef>

// Optimized Matmul using extreme register blocking and loop unrolling.
// We use an i-p-c-k loop order for maximum cache efficiency.
// We unroll the c-loop by 8.
// We use a manually unrolled 32-bit bit-extraction logic.

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
                    c3[k] += (p3 & mask) ? a_val : a_neg;
                    c4[k] += (p4 & mask) ? a_val : a_neg;
                    c5[k] += (p5 & mask) ? a_val : a_neg;
                    c6[k] += (p6 & mask) ? a_val : a_neg;
                    c7[k] += (p7 & mask) ? a_val : a_neg;
                }
            }

            // Handle remaining c chunks
            for (; c < K_ints; ++c) {
                const uint32_t packed = b_row[c];
                float* c_chunk = &c_row[c * 32];
                for (size_t k = 0; k < 3_bits_check_later_just_32; ++k) { 
                    // Wait, k < 32.
                }
            }
        }
    }
}
