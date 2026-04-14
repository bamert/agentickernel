#pragma once

using uint32_t = unsigned int;
using size_t   = unsigned long;

// Fast matmul using the identity:
//   C[i][j] = 2 * Σ_{p where B[p][j]=1} A[i][p] - Σ_p A[i][p]
// This version allocates a single reusable column‑accumulator buffer and
// clears it with memset each iteration to avoid repeated allocations.
void matmul(const float* A, const uint32_t* B, float* C,
            size_t M, size_t K) {
    const size_t K_ints = K / 32; // uint32 per row of B
    // Reusable buffer for Σ_{bit=1} A[i][p] per column.
    float* sum_one = new float[K];

    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        // Compute row sum.
        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) row_sum += a_row[p];

        // Clear accumulator efficiently.
        // Using memset works because zero pattern for float is all‑bits‑zero.
        memset(sum_one, 0, K * sizeof(float));

        // Accumulate contributions where B bit == 1.
        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            const uint32_t* b_row = B + p * K_ints;
            for (size_t blk = 0; blk < K_ints; ++blk) {
                uint32_t packed = b_row[blk];
                float* base = sum_one + blk * 32;
                while (packed) {
                    uint32_t t = __builtin_ctz(packed);
                    base[t] += a_val;
                    packed &= packed - 1; // clear lowest set bit
                }
            }
        }

        // Finalize row results.
        float* c_row = C + i * K;
        for (size_t j = 0; j < K; ++j) {
            c_row[j] = 2.0f * sum_one[j] - row_sum;
        }
    }
    delete[] sum_one;
}
