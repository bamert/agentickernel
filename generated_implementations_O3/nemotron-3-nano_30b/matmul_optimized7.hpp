#pragma once
#include <cstdint>
#include <cstddef>

// Optimized matmul: process four columns at a time with register accumulators.
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;  // K / 32

    for (size_t i = 0; i < M; ++i) {
        const float* A_i = A + i * K;   // Row i of A
        float*       C_i = C + i * K;   // Row i of C

        // Process each 32‑bit chunk of the packed B rows
        for (size_t chunk = 0; chunk < K_ints; ++chunk) {
            size_t col_base = chunk * 32;  // First column index of this chunk

            // Process the chunk in groups of 4 columns (offset steps of 4)
            #pragma unroll 8
            for (int offset = 0; offset < 32; offset += 4) {
                // Global column indices for the four elements we will update
                size_t col_idx0 = col_base + offset + 0;
                size_t col_idx1 = col_base + offset + 1;
                size_t col_idx2 = col_base + offset + 2;
                size_t col_idx3 = col_base + offset + 3;

                // Register accumulators for the four columns
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;

                // Loop over all rows p of the packed binary matrix
                for (size_t p = 0; p < K; ++p) {
                    float a_val = A_i[p];
                    const uint32_t* B_word = B + p * K_ints + chunk;  // Word holding bits for this chunk
                    uint32_t word = *B_word;

                    // Extract the four bits we care about
                    uint32_t bit0 = (word >> (offset + 0)) & 1u;
                    uint32_t bit1 = (word >> (offset + 1)) & 1u;
                    uint32_t bit2 = (word >> (offset + 2)) & 1u;
                    uint32_t bit3 = (word >> (offset + 3)) & 1u;

                    // Convert bits to sign (+1.0f or -1.0f)
                    float s0 = bit0 ? 1.0f : -1.0f;
                    float s1 = bit1 ? 1.0f : -1.0f;
                    float s2 = bit2 ? 1.0f : -1.0f;
                    float s3 = bit3 ? 1.0f : -1.0f;

                    // Multiply and accumulate
                    acc0 += a_val * s0;
                    acc1 += a_val * s1;
                    acc2 += a_val * s2;
                    acc3 += a_val * s3;
                }

                // Store the accumulated results back to C
                C_i[col_idx0] += acc0;
                C_i[col_idx1] += acc1;
                C_i[col_idx2] += acc2;
                C_i[col_idx3] += acc3;
            }
        }
    }
}