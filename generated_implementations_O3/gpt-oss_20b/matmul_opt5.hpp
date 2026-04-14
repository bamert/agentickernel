#pragma once
#include <cstddef>
#include <cstdint>

// Slightly more efficient implementation by processing two rows of C
// simultaneously to reduce loop overhead.  Works well when M is even
// (which is true for the test harness where M = 32).  It still uses
// the same cache‑friendly traversal as matmul_opt1.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    // Assumes M is even; if odd, process the last row separately.
    size_t i = 0;
    for (; i + 1 < M; i += 2) {
        float* rowC0 = &C[i * K];
        float* rowC1 = &C[(i + 1) * K];

        // Zero rows
        for (size_t j = 0; j < K; ++j) {
            rowC0[j] = 0.0f;
            rowC1[j] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) {
            float a0 = A[i * K + p];
            float a1 = A[(i + 1) * K + p];
            const uint32_t* packed_row = &B[p * K_ints];

            for (size_t blk = 0; blk < K_ints; ++blk) {
                uint32_t bits = packed_row[blk];
                size_t base = blk * 32;
                // Process 32 bits using a shift‑right per iteration.
                for (size_t bit = 0; bit < 32; ++bit) {
                    float sign = (bits & 1) ? 1.0f : -1.0f;
                    rowC0[base + bit] += a0 * sign;
                    rowC1[base + bit] += a1 * sign;
                    bits >>= 1;
                }
            }
        }
    }

    // Handle remaining odd row, if any.
    if (M & 1) {
        float* rowC = &C[i * K];
        for (size_t j = 0; j < K; ++j) rowC[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const uint32_t* packed_row = &B[p * K_ints];
            for (size_t blk = 0; blk < K_ints; ++blk) {
                uint32_t bits = packed_row[blk];
                size_t base = blk * 32;
                for (size_t bit = 0; bit < 32; ++bit) {
                    float sign = (bits & 1) ? 1.0f : -1.0f;
                    rowC[base + bit] += a_val * sign;
                    bits >>= 1;
                }
            }
        }
    }
}
