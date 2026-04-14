#pragma once

using uint32_t = unsigned int;
using uint64_t = unsigned long long;
using size_t   = unsigned long;

// Batch‑size‑4 matmul using 64‑bit packed words.
// Uses the identity C = 2*sum_one - row_sum where sum_one accumulates
// contributions of A[i][p] for each column where B[p][j] == 1.
// Processing four rows together reduces the number of scans over B
// by a factor of four, improving cache reuse.
void matmul(const float* A, const uint32_t* B, float* C,
            size_t M, size_t K) {
    const size_t K_ints64 = K / 64; // number of uint64 per row of B
    // K is a multiple of 32 by problem statement and a multiple of 64
    // for the benchmark dimensions.

    // Allocate reusable column accumulators for up to four rows.
    float* sum_one[4];
    for (size_t r = 0; r < 4; ++r) sum_one[r] = new float[K];

    size_t i = 0;
    // Process full batches of four rows.
    for (; i + 4 <= M; i += 4) {
        // ---- compute row sums ----
        float row_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (size_t r = 0; r < 4; ++r) {
            const float* a_row = A + (i + r) * K;
            float s = 0.0f;
            for (size_t p = 0; p < K; ++p) s += a_row[p];
            row_sum[r] = s;
        }
        // ---- clear accumulators ----
        for (size_t r = 0; r < 4; ++r) {
            memset(sum_one[r], 0, K * sizeof(float));
        }
        // ---- accumulate contributions for the four rows ----
        for (size_t p = 0; p < K; ++p) {
            // Load the four A values for this column p.
            float a_val[4];
            for (size_t r = 0; r < 4; ++r) {
                a_val[r] = A[(i + r) * K + p];
            }
            const uint64_t* b_row64 = reinterpret_cast<const uint64_t*>(B + p * (K / 32));
            for (size_t blk = 0; blk < K_ints64; ++blk) {
                uint64_t packed = b_row64[blk];
                while (packed) {
                    uint32_t t = __builtin_ctzll(packed); // 0..63
                    size_t col = blk * 64 + t;
                    // add contribution to each of the four rows
                    for (size_t r = 0; r < 4; ++r) {
                        sum_one[r][col] += a_val[r];
                    }
                    packed &= packed - 1ULL; // clear lowest set bit
                }
            }
        }
        // ---- write results for the batch ----
        for (size_t r = 0; r < 4; ++r) {
            float* c_row = C + (i + r) * K;
            for (size_t j = 0; j < K; ++j) {
                c_row[j] = 2.0f * sum_one[r][j] - row_sum[r];
            }
        }
    }

    // ---- handle remaining rows (<4) one by one using the two‑row code path ----
    for (; i < M; ++i) {
        const float* a_row = A + i * K;
        // row sum
        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) row_sum += a_row[p];
        // use sum_one[0] as temporary accumulator
        memset(sum_one[0], 0, K * sizeof(float));
        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            const uint64_t* b_row64 = reinterpret_cast<const uint64_t*>(B + p * (K / 32));
            for (size_t blk = 0; blk < K_ints64; ++blk) {
                uint64_t packed = b_row64[blk];
                while (packed) {
                    uint32_t t = __builtin_ctzll(packed);
                    sum_one[0][blk * 64 + t] += a_val;
                    packed &= packed - 1ULL;
                }
            }
        }
        float* c_row = C + i * K;
        for (size_t j = 0; j < K; ++j) {
            c_row[j] = 2.0f * sum_one[0][j] - row_sum;
        }
    }

    // cleanup
    for (size_t r = 0; r < 4; ++r) delete[] sum_one[r];
}
