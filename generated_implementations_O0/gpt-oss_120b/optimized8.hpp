#pragma once

using uint32_t = unsigned int;
using size_t   = unsigned long;

// Batched matmul (process two rows of A at once) using the identity:
//   C[i][j] = 2 * Σ_{p where B[p][j]=1} A[i][p] - Σ_p A[i][p]
// This reduces the number of scans over the packed matrix B roughly by half.
void matmul(const float* A, const uint32_t* B, float* C,
            size_t M, size_t K) {
    const size_t K_ints = K / 32; // number of uint32 per B row
    const size_t batch = 2;        // rows processed together

    // Allocate temporary accumulators for up to two rows.
    float* sum_one0 = new float[K];
    float* sum_one1 = new float[K];

    size_t i = 0;
    for (; i + 1 < M; i += batch) {
        const float* a_row0 = A + i * K;
        const float* a_row1 = A + (i + 1) * K;
        // Row sums.
        float row_sum0 = 0.0f, row_sum1 = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            row_sum0 += a_row0[p];
            row_sum1 += a_row1[p];
        }
        // Clear accumulators.
        for (size_t j = 0; j < K; ++j) {
            sum_one0[j] = 0.0f;
            sum_one1[j] = 0.0f;
        }
        // Accumulate contributions for both rows.
        for (size_t p = 0; p < K; ++p) {
            float a_val0 = a_row0[p];
            float a_val1 = a_row1[p];
            const uint32_t* b_row = B + p * K_ints;
            for (size_t blk = 0; blk < K_ints; ++blk) {
                uint32_t packed = b_row[blk];
                float* base0 = sum_one0 + blk * 32;
                float* base1 = sum_one1 + blk * 32;
                while (packed) {
                    uint32_t t = __builtin_ctz(packed);
                    base0[t] += a_val0;
                    base1[t] += a_val1;
                    packed &= packed - 1; // clear lowest set bit
                }
            }
        }
        // Write results for the two rows.
        float* c_row0 = C + i * K;
        float* c_row1 = C + (i + 1) * K;
        for (size_t j = 0; j < K; ++j) {
            c_row0[j] = 2.0f * sum_one0[j] - row_sum0;
            c_row1[j] = 2.0f * sum_one1[j] - row_sum1;
        }
    }
    // Handle possible leftover row when M is odd.
    if (i < M) {
        const float* a_row = A + i * K;
        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) row_sum += a_row[p];
        for (size_t j = 0; j < K; ++j) sum_one0[j] = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            const uint32_t* b_row = B + p * K_ints;
            for (size_t blk = 0; blk < K_ints; ++blk) {
                uint32_t packed = b_row[blk];
                float* base = sum_one0 + blk * 32;
                while (packed) {
                    uint32_t t = __builtin_ctz(packed);
                    base[t] += a_val;
                    packed &= packed - 1;
                }
            }
        }
        float* c_row = C + i * K;
        for (size_t j = 0; j < K; ++j) {
            c_row[j] = 2.0f * sum_one0[j] - row_sum;
        }
    }
    delete[] sum_one0;
    delete[] sum_one1;
}
