#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5; // K / 32

    // Outer loop: each row p of the packed binary matrix B
    for (size_t p = 0; p < K; ++p) {
        const uint32_t* B_row = B + p * K_ints; // start of row p

        // Process each 32‑bit chunk of this row
        for (size_t chunk = 0; chunk < K_ints; ++chunk) {
            uint32_t word = B_row[chunk];
            size_t col_base = chunk * 32; // first column index of this chunk

            // Process 4 columns at a time
            #pragma unroll 8
            for (int offset = 0; offset < 32; offset += 4) {
                // Global column indices for the four elements we will update
                size_t idx0 = col_base + offset + 0;
                size_t idx1 = col_base + offset + 1;
                size_t idx2 = col_base + offset + 2;
                size_t idx3 = col_base + offset + 3;

                // Process all rows i of A (i = output row)
                #pragma unroll 4
                for (size_t i = 0; i < M; ++i) {
                    float a_val = A[i * K + p]; // scalar from A

                    // Extract the 4 bits that correspond to the four columns we are updating
                    uint32_t nibble = (word >> offset) & 0xF;

                    // Compute sign for each bit: +1 if bit==1, -1 otherwise, branchless
                    float s0 = -1.0f + 2.0f * ((nibble >> 0) & 1u);
                    float s1 = -1.0f + 2.0f * ((nibble >> 1) & 1u);
                    float s2 = -1.0f + 2.0f * ((nibble >> 2) & 1u);
                    float s3 = -1.0f + 2.0f * ((nibble >> 3) & 1u);

                    // Directly update C
                    C[i * K + idx0] += a_val * s0;
                    C[i * K + idx1] += a_val * s1;
                    C[i * K + idx2] += a_val * s2;
                    C[i * K + idx3] += a_val * s3;
                }
            }
        }
    }
}