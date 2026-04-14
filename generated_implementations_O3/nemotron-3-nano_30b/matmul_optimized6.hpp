#pragma once
#include <cstdint>
#include <cstddef>

// Optimized matmul: compute sign pattern once per packed row, then update C
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K >> 5;  // K / 32

    // Iterate over each row p of the packed binary matrix B
    for (size_t p = 0; p < K; ++p) {
        const uint32_t* B_row = B + p * K_ints;  // Packed bits for row p

        // Process the row in 32‑bit chunks
        for (size_t chunk = 0; chunk < K_ints; ++chunk) {
            uint32_t word = B_row[chunk];

            // Process 4 columns at a time
            #pragma unroll 8
            for (size_t offset = 0; offset < 32; offset += 4) {
                // Extract four bits
                uint32_t bit0 = (word >> offset) & 1u;
                uint32_t bit1 = (word >> (offset + 1)) & 1u;
                uint32_t bit2 = (word >> (offset + 2)) & 1u;
                uint32_t bit3 = (word >> (offset + 3)) & 1u;

                // Convert bits to +1.0f or -1.0f
                float s0 = bit0 ? 1.0f : -1.0f;
                float s1 = bit1 ? 1.0f : -1.0f;
                float s2 = bit2 ? 1.0f : -1.0f;
                float s3 = bit3 ? 1.0f : -1.0f;

                // Global column indices for the four elements
                size_t base = chunk * 32 + offset;

                // Update all M rows for these four columns
                #pragma unroll 4
                for (size_t i = 0; i < M; ++i) {
                    float a_val = A[i * K + p];
                    C[i * K + base + 0] += a_val * s0;
                    C[i * K + base + 1] += a_val * s1;
                    C[i * K + base + 2] += a_val * s2;
                    C[i * K + base + 3] += a_val * s3;
                }
            }
        }
    }
}