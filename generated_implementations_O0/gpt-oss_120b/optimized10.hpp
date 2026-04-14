#pragma once

using uint32_t = unsigned int;
using size_t   = unsigned long;

// Batched matrix multiplication processing up to 8 rows of A at once.
// This halves (or more) the number of scans over the packed matrix B.
// The core idea is the same identity C = 2*sum_one - row_sum.
// For each batch we allocate one accumulator per row and update them
// together for each set bit in B.
void matmul(const float* A, const uint32_t* B, float* C,
            size_t M, size_t K) {
    const size_t K_ints = K / 32;           // number of uint32 per row of B
    const size_t BATCH   = 8;               // rows processed together (max)

    // Allocate accumulators for the maximum batch size.
    float* sum_one[BATCH];
    for (size_t b = 0; b < BATCH; ++b) {
        sum_one[b] = new float[K];
    }

    size_t i = 0;
    for (; i + BATCH <= M; i += BATCH) {
        // ----- compute row sums -----
        float row_sum[BATCH];
        for (size_t r = 0; r < BATCH; ++r) {
            const float* a_row = A + (i + r) * K;
            float s = 0.0f;
            for (size_t p = 0; p < K; ++p) s += a_row[p];
            row_sum[r] = s;
        }
        // ----- clear accumulators -----
        for (size_t r = 0; r < BATCH; ++r) {
            for (size_t j = 0; j < K; ++j) sum_one[r][j] = 0.0f;
        }
        // ----- accumulate contributions -----
        for (size_t p = 0; p < K; ++p) {
            // load A values for all rows in this batch at position p
            float a_val[BATCH];
            for (size_t r = 0; r < BATCH; ++r) {
                a_val[r] = A[(i + r) * K + p];
            }
            const uint32_t* b_row = B + p * K_ints;
            for (size_t blk = 0; blk < K_ints; ++blk) {
                uint32_t packed = b_row[blk];
                while (packed) {
                    uint32_t t = __builtin_ctz(packed);
                    // update each row's accumulator for column (blk*32 + t)
                    size_t col = blk * 32 + t;
                    for (size_t r = 0; r < BATCH; ++r) {
                        sum_one[r][col] += a_val[r];
                    }
                    packed &= packed - 1; // clear lowest set bit
                }
            }
        }
        // ----- write results -----
        for (size_t r = 0; r < BATCH; ++r) {
            float* c_row = C + (i + r) * K;
            for (size_t j = 0; j < K; ++j) {
                c_row[j] = 2.0f * sum_one[r][j] - row_sum[r];
            }
        }
    }

    // ----- handle remaining rows (less than a full batch) -----
    for (; i < M; ++i) {
        const float* a_row = A + i * K;
        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) row_sum += a_row[p];
        float* acc = sum_one[0];
        for (size_t j = 0; j < K; ++j) acc[j] = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            const uint32_t* b_row = B + p * K_ints;
            for (size_t blk = 0; blk < K_ints; ++blk) {
                uint32_t packed = b_row[blk];
                while (packed) {
                    uint32_t t = __builtin_ctz(packed);
                    acc[blk * 32 + t] += a_val;
                    packed &= packed - 1;
                }
            }
        }
        float* c_row = C + i * K;
        for (size_t j = 0; j < K; ++j) {
            c_row[j] = 2.0f * acc[j] - row_sum;
        }
    }

    // cleanup
    for (size_t b = 0; b < BATCH; ++b) delete[] sum_one[b];
}
