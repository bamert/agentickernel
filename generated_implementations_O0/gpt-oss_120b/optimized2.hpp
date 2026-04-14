#pragma once
#include <cstddef>
#include <cstdint>

// Further optimized matmul using the "2*sum_one - row_sum" identity.
// Avoids per‑row dynamic allocation by reusing a single buffer.
// Uses __builtin_ctz to iterate over set bits only.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32; // number of uint32 per row of B
    // Allocate a reusable accumulator for the per‑column sums of bits==1.
    float* sum_one = new float[K];

    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        // Compute total sum of the current row of A.
        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            row_sum += a_row[p];
        }
        // Clear accumulator.
        for (size_t j = 0; j < K; ++j) {
            sum_one[j] = 0.0f;
        }
        // Accumulate contributions where B bit == 1.
        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            const uint32_t* b_row = B + p * K_ints;
            for (size_t block = 0; block < K_ints; ++block) {
                uint32_t packed = b_row[block];
                while (packed) {
                    uint32_t t = __builtin_ctz(packed);
                    size_t col = block * 32 + t;
                    sum_one[col] += a_val;
                    packed &= packed - 1; // clear lowest set bit
                }
            }
        }
        // Write final results for this row.
        float* c_row = C + i * K;
        for (size_t j = 0; j < K; ++j) {
            c_row[j] = 2.0f * sum_one[j] - row_sum;
        }
    }
    delete[] sum_one;
}
