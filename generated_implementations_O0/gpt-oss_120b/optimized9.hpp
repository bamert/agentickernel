#pragma once

using uint32_t = unsigned int;
using size_t   = unsigned long;

// Batched matrix multiplication (process up to 4 rows of A at once).
// Uses the identity C = 2*sum_one - row_sum where sum_one accumulates
// contributions of A[i][p] for bits==1 in B.
// By handling several rows together we scan the packed matrix B only once
// per batch, reducing memory traffic.
void matmul(const float* A, const uint32_t* B, float* C,
            size_t M, size_t K) {
    const size_t K_ints = K / 32; // uint32 per row of B
    const size_t BATCH   = 4;      // number of rows processed together

    // Allocate per‑row accumulators (maximum batch size).
    float* sum_one[ BATCH ];
    for (size_t b = 0; b < BATCH; ++b) {
        sum_one[b] = new float[K];
    }

    size_t i = 0;
    for (; i + BATCH <= M; i += BATCH) {
        // ----- compute row sums -----
        float row_sum[ BATCH ];
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
        // ----- accumulate contributions for all rows in the batch -----
        for (size_t p = 0; p < K; ++p) {
            const uint32_t* b_row = B + p * K_ints;
            // load the A values for this p for all rows in the batch
            float a_val[ BATCH ];
            for (size_t r = 0; r < BATCH; ++r) {
                a_val[r] = A[(i + r) * K + p];
            }
            for (size_t blk = 0; blk < K_ints; ++blk) {
                uint32_t packed = b_row[blk];
                // process each set bit and add the corresponding a_val to each row's accumulator
                while (packed) {
                    uint32_t t = __builtin_ctz(packed);
                    for (size_t r = 0; r < BATCH; ++r) {
                        sum_one[r][blk * 32 + t] += a_val[r];
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

    // ----- handle remaining rows (if M not multiple of BATCH) -----
    for (; i < M; ++i) {
        const float* a_row = A + i * K;
        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) row_sum += a_row[p];
        float* sum_one0 = sum_one[0];
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

    // cleanup
    for (size_t b = 0; b < BATCH; ++b) delete[] sum_one[b];
}
