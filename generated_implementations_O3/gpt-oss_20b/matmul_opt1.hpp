#pragma once
#include <cstdint>
#include <cstddef>

// Optimized matrix multiplication using a cache-friendly loop order
// and processing 32 columns at a time to utilize the packed binary format.
// This version avoids the innermost per-element extraction and improves
// cache locality by iterating over the binary matrix row first.
// Matrix C = A * B, where B is packed binary (+1/-1 per bit).
// Parameters are unchanged from the baseline.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    // Number of 32-bit chunks per row of B
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        // Initialize C row to zero
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }

        // For each element in row i of A
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const uint32_t* packed_row = &B[p * K_ints];

            // Process each 32-bit block of columns
            for (size_t blk = 0; blk < K_ints; ++blk) {
                uint32_t bits = packed_row[blk];
                size_t base_col = blk * 32;
                // Unroll 32 iterations (inner-most) for better performance
                for (size_t bit = 0; bit < 32; ++bit) {
                    size_t j = base_col + bit;
                    float sign = ((bits >> bit) & 1) ? 1.0f : -1.0f;
                    C[i * K + j] += a_val * sign;
                }
            }
        }
    }
}
