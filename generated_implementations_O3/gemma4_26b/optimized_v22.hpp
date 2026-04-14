#pragma once
#include <cstdint>
#include <cstddef>

// Optimized Matmul using extreme register blocking and loop unrolling.
// We use an i-p-c-k loop order for maximum cache efficiency.
// We unroll the c-loop by 4 to handle 128 columns of B at once.
// We use a manually unrolled 32-bit bit-extraction logic.
// This version avoids the macro-based mistakes of previous attempts.

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
            // Unroll c-loop by 4
            for (; c + 3 < K_ints; c += 4) {
                const uint32_t p0 = b_row[c];
                const uint32_t p1 = b_row[c + 1];
                const uint32_t p2 = b_row[c + 2];
                const uint32_t p3 = b_row[c + 3];

                float* c0 = &c_row[c * 32];
                float* c1 = &c_row[(c + 1) * 32];
                float* c2 = &c_row[(c + 2) * 32];
                float* c3 = &c_row[(c + 3) * 32];

                // Hand-unroll 32 bits for each chunk to maximize ILP and avoid branches
                // We use a loop for the bits within the macro-like structure to keep it clean
                // but since we want speed, we will just use a small loop for the bit-processing
                // inside the c-loop, which the compiler can unroll.
                for (size_t k = 0; k < 32; ++k) {
                    uint32_t mask = (1U << k);
                    c0[k] += (p0 & mask) ? a_val : a_neg;
                    c1[k] += (p1 & mask) ? a_val : a_neg;
                    c2[k] += (p2 & mask) ? a_val : a_neg;
                    c3[k] += (p3 & mask) ? a_val : a_neg;
                }
            }

            // Handle remaining c chunks
            for (; c < K_ints; ++c) {
                const uint32_t packed = b_row[c];
                float* c_chunk = &c_row[c * 32];
                for (size_t k = 0; k < 31; ++k) { // 32 bits
                     uint32_t mask = (1U << k);
                     c_chunk[k] += (packed & mask) ? a_val : a_neg;
                }
                // The last bit
                c_chunk[31] += (packed & (1U << 31)) ? a_val : a_neg;
            }
        }
    }
}
