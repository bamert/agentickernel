#pragma once
#include <cstddef>
#include <cstdint>

// Optimized matrix multiplication using bit counting trick.
// Computes C = A * B where B is packed binary (+1/-1) matrix.
// For each row i of A, we precompute the total sum of the row.
// Then each column j of B: result = 2 * sum_{p where bit=1} A[i][p] - total_sum.
// This reduces the per-element sign multiplication to a single subtraction.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32; // number of uint32 per row of B
    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        // compute total sum of the row once
        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            row_sum += a_row[p];
        }
        // For each column j compute the contribution of bits == 1
        for (size_t j = 0; j < K; ++j) {
            float sum_one = 0.0f;
            // iterate over packed ints of column j (i.e., rows of B)
            for (size_t block = 0; block < K_ints; ++block) {
                uint32_t packed = B[block + j * K_ints]; // WRONG indexing? Need correct.
            }
        }
    }
}
