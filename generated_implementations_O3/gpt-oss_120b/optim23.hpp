#pragma once

#include <cstddef>
#include <cstdint>

// Optimised matrix multiply using the identity sign = 2*bit-1.
// For each row of A we pre‑compute the row sum and initialise the output row with -row_sum.
// Then we only add 2*a_val for columns where the corresponding bit in B is 1.
// This eliminates the sign lookup table and reduces the number of arithmetic operations.

void matmul(const float* __restrict A,
            const uint32_t* __restrict B,
            float* __restrict C,
            size_t M,
            size_t K)
{
    const size_t K_ints = K / 32; // number of 32‑bit words per row of B

    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;
        // 1. Compute row sum of A and initialise output row with -row_sum.
        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            row_sum += a_row[p];
        }
        for (size_t j = 0; j < K; ++j) {
            c_row[j] = -row_sum;
        }
        // 2. For each element of A, add 2*a_val to columns where B has a set bit.
        for (size_t p = 0; p < K; ++p) {
            const float inc = 2.0f * a_row[p];
            const uint32_t* b_ptr = B + p * K_ints; // start of this row in B
            size_t block = 0;
            size_t base = 0;
            // Process each 32‑bit block (32 columns).
            for (; block < K_ints; ++block, base += 32) {
                uint32_t bits = b_ptr[block];
                // Iterate over set bits only.
                while (bits) {
                    unsigned idx = __builtin_ctz(bits); // position of lowest set bit (0‑31)
                    bits &= bits - 1;                    // clear that bit
                    c_row[base + idx] += inc;
                }
            }
        }
    }
}
