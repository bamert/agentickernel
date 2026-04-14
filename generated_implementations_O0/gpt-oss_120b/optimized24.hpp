#pragma once

using uint32_t = unsigned int;
using uint64_t = unsigned long long;
using size_t   = unsigned long;

// Batch size 4 matmul using 64‑bit packed B.
// Identity: C[i][j] = 2*sum_one - row_sum, where sum_one accumulates
// A[i][p] for each column j such that B[p][j] == 1.
// This version scans the packed matrix B only once per batch of four rows
// and processes 64 columns at a time using a uint64_t word.
void matmul(const float* A, const uint32_t* B, float* C,
            size_t M, size_t K) {
    const size_t ints_per_row = K / 32; // uint32 per row of B (for pointer arithmetic)
    const size_t K_ints64 = K / 64;     // uint64 words per row of B
    const size_t BATCH = 4;            // rows processed together

    // Allocate reusable column accumulators for up to BATCH rows.
    float* sum_one[BATCH];
    for (size_t r = 0; r < BATCH; ++r) sum_one[r] = new float[K];

    size_t i = 0;
    // Process full batches of four rows.
    for (; i + BATCH <= M; i += BATCH) {
        // ---- compute row sums ----
        float row_sum[BATCH];
        for (size_t r = 0; r < BATCH; ++r) {
            const float* a_row = A + (i + r) * K;
            float s = 0.0f;
            for (size_t p = 0; p < K; ++p) s += a_row[p];
            row_sum[r] = s;
        }
        // ---- clear accumulators ----
        for (size_t r = 0; r < BATCH; ++r) {
            for (size_t j = 0; j < K; ++j) sum_one[r][j] = 0.0f;
        }
        // ---- accumulate contributions ----
        for (size_t p = 0; p < K; ++p) {
            // Load the four A values for this column p.
            float a_val[BATCH];
            for (size_t r = 0; r < BATCH; ++r) {
                a_val[r] = A[(i + r) * K + p];
            }
            const uint64_t* b_row64 = reinterpret_cast<const uint64_t*>(B + p * ints_per_row);
            for (size_t blk = 0; blk < K_ints64; ++blk) {
                uint64_t packed = b_row64[blk];
                while (packed) {
                    uint32_t t = __builtin_ctzll(packed); // bit position 0..63
                    size_t col = blk * 64 + t;
                    for (size_t r = 0; r < BATCH; ++r) {
                        sum_one[r][col] += a_val[r];
                    }
                    packed &= packed - 1ULL; // clear lowest set bit
                }
            }
        }
        // ---- write results for the batch ----
        for (size_t r = 0; r < BATCH; ++r) {
            float* c_row = C + (i + r) * K;
            for (size_t j = 0; j < K; ++j) {
                c_row[j] = 2.0f * sum_one[r][j] - row_sum[r];
            }
        }
    }

    // ---- handle remaining rows (less than a full batch) ----
    for (; i < M; ++i) {
        const float* a_row = A + i * K;
        // row sum
        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) row_sum += a_row[p];
        // use sum_one[0] as temporary accumulator
        float* acc = sum_one[0];
        for (size_t j = 0; j < K; ++j) acc[j] = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            const uint64_t* b_row64 = reinterpret_cast<const uint64_t*>(B + p * ints_per_row);
            for (size_t blk = 0; blk < K_ints64; ++blk) {
                uint64_t packed = b_row64[blk];
                while (packed) {
                    uint32_t t = __builtin_ctzll(packed);
                    acc[blk * 64 + t] += a_val;
                    packed &= packed - 1ULL;
                }
            }
        }
        float* c_row = C + i * K;
        for (size_t j = 0; j < K; ++j) {
            c_row[j] = 2.0f * acc[j] - row_sum;
        }
    }

    // cleanup
    for (size_t r = 0; r < BATCH; ++r) delete[] sum_one[r];
}
