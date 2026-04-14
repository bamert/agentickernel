#pragma once
#include <cstddef>
#include <cstdint>

// Optimized matmul using the identity:
//   C[i][j] = 2 * Σ_{p where B[p][j]=1} A[i][p] - Σ_p A[i][p]
// This avoids per‑element sign multiplication. The implementation
// reuses a single temporary buffer for the per‑column "sum_one"
// accumulator to reduce allocation overhead and uses a fast
// "extract‑lowest‑set‑bit" loop.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;               // uint32s per row of B
    float* sum_one = new float[K];              // reusable accumulator

    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        // Compute total sum of this row once.
        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) row_sum += a_row[p];

        // Zero accumulator.
        for (size_t j = 0; j < K; ++j) sum_one[j] = 0.0f;

        // Accumulate a_row[p] into each column where B's bit is 1.
        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            const uint32_t* b_row = B + p * K_ints;
            for (size_t block = 0; block < K_ints; ++block) {
                uint32_t packed = b_row[block];
                while (packed) {
                    uint32_t t = __builtin_ctz(packed);          // position of LS set bit
                    size_t col = block * 32 + t;                // column index
                    sum_one[col] += a_val;
                    packed &= packed - 1;                       // clear LS set bit
                }
            }
        }

        // Write final results for the row.
        float* c_row = C + i * K;
        for (size_t j = 0; j < K; ++j) {
            c_row[j] = 2.0f * sum_one[j] - row_sum;
        }
    }
    delete[] sum_one;
}
