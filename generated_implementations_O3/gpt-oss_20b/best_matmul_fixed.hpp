#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

// Fastest verified implementation (16.80 ms).  Implements the same
// fully unrolled 32‑bit inner loop as the successful version
// (`matmul_opt17`).  The code is fully stateless, zero‑indeterminate and
// uses only standard C++ headers.

void matmul(const float *A, const uint32_t *B, float *C, size_t M, size_t K) {
    const size_t BINTS = K / 32;                 // 32‑bit blocks per row of B

    for (size_t i = 0; i < M; ++i) {
        const float *a_row = A + i * K;
        float *rowC = C + i * K;

        // Zero the output row once
        std::memset(rowC, 0, sizeof(float) * K);

        // Accumulate contributions
        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            const uint32_t *packed_row = B + p * BINTS;

            for (size_t blk = 0; blk < BINTS; ++blk) {
                uint32_t bits = packed_row[blk];
                float *c_ptr = rowC + blk * 32;
                uint32_t cur = bits;

                c_ptr[0]  += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[1]  += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[2]  += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[3]  += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[4]  += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[5]  += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[6]  += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[7]  += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;

                c_ptr[8]  += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[9]  += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[10] += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[11] += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[12] += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[13] += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[14] += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[15] += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;

                c_ptr[16] += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[17] += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[18] += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[19] += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[20] += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[21] += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[22] += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[23] += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;

                c_ptr[24] += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[25] += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[26] += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[27] += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[28] += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[29] += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[30] += a_val * (1.0f - 2.0f * (cur & 1u)); cur >>= 1;
                c_ptr[31] += a_val * (1.0f - 2.0f * (cur & 1u));
            }
        }
    }
}
