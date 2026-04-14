#pragma once
#include <cstddef>
#include <cstdint>

// Optimized matmul using the identity:
//   sum = Σ a_p * (2*bit - 1) = 2*Σ_{bit=1} a_p - Σ a_p
// This reduces the inner loop to only additions for bits == 1.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32; // number of uint32 per row of B
    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        // Compute total sum of this row once.
        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            row_sum += a_row[p];
        }
        // Allocate temporary accumulator for Σ_{bit=1} a_p for each column.
        float* sum_one = new float[K];
        for (size_t j = 0; j < K; ++j) sum_one[j] = 0.0f;
        // Iterate over each p (row of B) and add a_row[p] to columns where bit == 1.
        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            const uint32_t* b_row = B + p * K_ints;
            for (size_t block = 0; block < K_ints; ++block) {
                uint32_t packed = b_row[block];
                // Process set bits only.
                while (packed) {
                    uint32_t t = __builtin_ctz(packed); // index of least‑significant set bit
                    size_t col = block * 32 + t;
                    sum_one[col] += a_val;
                    packed &= packed - 1; // clear lowest set bit
                }
            }
        }
        // Produce final results.
        float* c_row = C + i * K;
        for (size_t j = 0; j < K; ++j) {
            c_row[j] = 2.0f * sum_one[j] - row_sum;
        }
        delete[] sum_one;
    }
}
