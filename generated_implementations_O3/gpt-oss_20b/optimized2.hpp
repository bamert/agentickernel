#pragma once
#include <cstdint>
#include <cstddef>

// Optimized matmul using per-row block accumulation to reduce C memory traffic.
// For each row of A, we process columns in blocks of 8. For each block we
// accumulate partial sums in a local array of 8 floats, then write them back
// to C after processing all rows of B. This reduces the number of writes to C.
// The bit extraction for the packed binary matrix B remains scalar but
// happens only once per (p, column) pair, not once per (p, column, row).

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    const size_t block = 8;          // Block size (must divide K for simplicity)
    const size_t blocks_per_row = (K + block - 1) / block;

    for (size_t i = 0; i < M; ++i) {
        for (size_t blk_idx = 0; blk_idx < blocks_per_row; ++blk_idx) {
            const size_t start_col = blk_idx * block;
            const size_t end_col = (start_col + block <= K) ? (start_col + block) : K;
            float sums[8] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};

            // For each shared dimension value p
            for (size_t p = 0; p < K; ++p) {
                float a_val = A[i * K + p];
                const uint32_t* B_row = &B[p * K_ints];

                // Extract bits for columns in this block
                for (size_t idx = 0; idx < (end_col - start_col); ++idx) {
                    size_t j = start_col + idx;
                    const uint32_t packed = B_row[j / 32];
                    const uint32_t bit = (packed >> (j % 32)) & 1U;
                    const float sign = bit ? 1.0f : -1.0f;
                    sums[idx] += a_val * sign;
                }
            }

            // Write results back to C
            for (size_t idx = 0; idx < (end_col - start_col); ++idx) {
                C[i * K + start_col + idx] = sums[idx];
            }
        }
    }
}
