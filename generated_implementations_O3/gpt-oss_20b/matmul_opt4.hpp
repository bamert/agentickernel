#pragma once
#include <cstdint>
#include <cstddef>

// Slight improvement over matmul_opt1: use a single shift per inner loop
// to avoid repeated (bits >> bit) operations.  Also use a pointer to the
// destination row to reduce address calculations.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* rowC = &C[i * K];
        // Zero the row before accumulation
        for (size_t j = 0; j < K; ++j) {
            rowC[j] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const uint32_t* packed_row = &B[p * K_ints];

            for (size_t blk = 0; blk < K_ints; ++blk) {
                uint32_t bits = packed_row[blk];
                size_t base = blk * 32;
                uint32_t bits_local = bits;
                for (size_t bit = 0; bit < 32; ++bit) {
                    float sign = (bits_local & 1) ? 1.0f : -1.0f;
                    rowC[base + bit] += a_val * sign;
                    bits_local >>= 1;
                }
            }
        }
    }
}
