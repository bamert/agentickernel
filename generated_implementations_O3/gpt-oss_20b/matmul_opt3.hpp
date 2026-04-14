#pragma once
#include <cstdint>
#include <cstddef>

// Slightly improved cache-aware version using pointer arithmetic for the
// destination matrix C.  The overall algorithm is the same as matmul_opt1.
// This change removes an array subscript per element and slightly reduces
// instruction count.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        // Pointer to the start of this row of C
        float* rowC = &C[i * K];
        // Zero the row
        for (size_t j = 0; j < K; ++j) {
            rowC[j] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const uint32_t* packed_row = &B[p * K_ints];

            for (size_t blk = 0; blk < K_ints; ++blk) {
                uint32_t bits = packed_row[blk];
                size_t base_col = blk * 32;
                // Pointer to the first element of this block in rowC
                float* block_ptr = &rowC[base_col];

                for (size_t bit = 0; bit < 32; ++bit) {
                    float sign = ((bits >> bit) & 1) ? 1.0f : -1.0f;
                    block_ptr[bit] += a_val * sign;
                }
            }
        }
    }
}
