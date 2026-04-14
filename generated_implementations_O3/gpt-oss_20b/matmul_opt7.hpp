#pragma once
#include <cstdint>
#include <cstddef>

// Optimised for minimal branch/predicate overhead – we shift the bits
// each iteration and use mask to decide the sign.  The inner loop is
// deliberately kept extremely small.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* rowC = C + i * K;
        // zero the row
        for (size_t j = 0; j < K; ++j) rowC[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            const uint32_t* packed_row = B + p * K_ints;

            for (size_t blk = 0; blk < K_ints; ++blk) {
                uint32_t bits = packed_row[blk];
                float* c_ptr = rowC + blk * 32;
                uint32_t cur = bits;
                for (size_t bit = 0; bit < 32; ++bit) {
                    float sign = ((cur & 1u) ? 1.0f : -1.0f);
                    c_ptr[bit] += a_val * sign;
                    cur >>= 1;
                }
            }
        }
    }
}
