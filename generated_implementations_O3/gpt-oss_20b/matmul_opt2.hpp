#pragma once
#include <cstdint>
#include <cstddef>

// Further optimizations : use a cache-friendly layout and reduce inner loop overhead
// by unrolling the 32-bit block into chunks of 4 bits.
// The function signature remains identical to the baseline.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* row = &C[i * K]; // pointer to the start of the current row of C

        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const uint32_t* packed_row = &B[p * K_ints];

            for (size_t blk = 0; blk < K_ints; ++blk) {
                uint32_t bits = packed_row[blk];
                size_t base_col = blk * 32;

                // Process 8 groups of 4 bits each (32 bits total)
                for (size_t offset = 0; offset < 32; offset += 4) {
                    uint32_t sub = (bits >> offset) & 0xFu;
                    // Sign for each of the 4 bits; 1.0f for bit=1, -1.0f for bit=0
                    float s0 = ((sub >> 0) & 1u) ? 1.0f : -1.0f;
                    float s1 = ((sub >> 1) & 1u) ? 1.0f : -1.0f;
                    float s2 = ((sub >> 2) & 1u) ? 1.0f : -1.0f;
                    float s3 = ((sub >> 3) & 1u) ? 1.0f : -1.0f;

                    row[base_col + offset + 0] += a_val * s0;
                    row[base_col + offset + 1] += a_val * s1;
                    row[base_col + offset + 2] += a_val * s2;
                    row[base_col + offset + 3] += a_val * s3;
                }
            }
        }
    }
}
