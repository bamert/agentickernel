#pragma once

using uint32_t = unsigned int;
using uint64_t = unsigned long long;
using size_t   = unsigned long;

// Two‑row batched matmul with 64‑bit packed B, prefetching and __restrict
// Identity: C = 2*sum_one - row_sum.
void matmul(const float* __restrict A, const uint32_t* __restrict B, float* __restrict C,
            size_t M, size_t K) {
    const size_t K_ints64 = K / 64;            // uint64 per row of B
    const size_t ints_per_row = K / 32;        // uint32 per row of B (used for pointer arithmetic)
    const size_t prefetch_dist = 16;           // rows ahead to prefetch

    // Reusable per‑row column accumulators for the two rows.
    float* sum_one0 = new float[K];
    float* sum_one1 = new float[K];

    size_t i = 0;
    for (; i + 1 < M; i += 2) {
        const float* a_row0 = A + i * K;
        const float* a_row1 = A + (i + 1) * K;

        // ---- compute row sums ----
        float row_sum0 = 0.0f, row_sum1 = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            row_sum0 += a_row0[p];
            row_sum1 += a_row1[p];
        }

        // ---- clear accumulators ----
        memset(sum_one0, 0, K * sizeof(float));
        memset(sum_one1, 0, K * sizeof(float));

        // ---- accumulate contributions ----
        for (size_t p = 0; p < K; ++p) {
            // Prefetch future B rows to improve cache utilization.
            if (p + prefetch_dist < K) {
                __builtin_prefetch(B + (p + prefetch_dist) * ints_per_row, 0, 1);
            }
            float a_val0 = a_row0[p];
            float a_val1 = a_row1[p];
            const uint64_t* b_row64 = reinterpret_cast<const uint64_t*>(B + p * ints_per_row);
            for (size_t blk = 0; blk < K_ints64; ++blk) {
                uint64_t packed = b_row64[blk];
                float* base0 = sum_one0 + blk * 64;
                float* base1 = sum_one1 + blk * 64;
                while (packed) {
                    uint32_t t = __builtin_ctzll(packed); // index of set bit (0..63)
                    base0[t] += a_val0;
                    base1[t] += a_val1;
                    packed &= packed - 1ULL; // clear lowest set bit
                }
            }
        }

        // ---- write results ----
        float* c_row0 = C + i * K;
        float* c_row1 = C + (i + 1) * K;
        for (size_t j = 0; j < K; ++j) {
            c_row0[j] = 2.0f * sum_one0[j] - row_sum0;
            c_row1[j] = 2.0f * sum_one1[j] - row_sum1;
        }
    }

    // ---- handle trailing row if M is odd ----
    if (i < M) {
        const float* a_row = A + i * K;
        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) row_sum += a_row[p];
        memset(sum_one0, 0, K * sizeof(float));
        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            const uint64_t* b_row64 = reinterpret_cast<const uint64_t*>(B + p * ints_per_row);
            for (size_t blk = 0; blk < K_ints64; ++blk) {
                uint64_t packed = b_row64[blk];
                float* base = sum_one0 + blk * 64;
                while (packed) {
                    uint32_t t = __builtin_ctzll(packed);
                    base[t] += a_val;
                    packed &= packed - 1ULL;
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
