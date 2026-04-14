#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;               // K / 32

    // Process each output row i
    for (size_t i = 0; i < M; ++i) {
        // Temporary accumulator for this row (K ≤ 4096 for the test)
        float accum[4096];
        for (size_t j = 0; j < K; ++j) accum[j] = 0.0f;

        const float* A_row = A + i * K;   // row i of A
        float*       C_row = C + i * K;   // row i of C

        // Process each row p of the packed binary matrix B
        for (size_t p = 0; p < K; ++p) {
            float a_val = A_row[p];
            const uint32_t* B_row = B + p * K_ints;   // start of row p in packed B

            // Process the row in 32‑bit chunks
            for (size_t chunk = 0; chunk < K_ints; ++chunk) {
                uint32_t word = B_row[chunk];
                size_t   col_base = chunk * 32;        // first column index of this chunk

                // Process four columns at a time
                #pragma unroll 8
                for (int offset = 0; offset < 32; offset += 4) {
                    // Extract the 4‑bit nibble that encodes the signs for these columns
                    uint32_t nibble = (word >> offset) & 0xF;

                    // Convert each bit of the nibble to +1.0f or -1.0f without branching
                    // sign = -1.0f + 2.0f * ((nibble >> t) & 1u);
                    float s0 = -1.0f + 2.0f * static_cast<float>((nibble >> 0) & 1u);
                    float s1 = -1.0f + 2.0f * static_cast<float>((nibble >> 1) & 1u);
                    float s2 = -1.0f + 2.0f * static_cast<float>((nibble >> 2) & 1u);
                    float s3 = -1.0f + 2.0f * static_cast<float>((nibble >> 3) & 1u);

                    // Global column indices for the four elements we will update
                    size_t idx0 = col_base + offset + 0;
                    size_t idx1 = col_base + offset + 1;
                    size_t idx2 = col_base + offset + 2;
                    size_t idx3 = col_base + offset + 3;

                    // Accumulate the contribution of a_val and the four signs
                    accum[idx0] += a_val * s0;
                    accum[idx1] += a_val * s1;
                    accum[idx2] += a_val * s2;
                    accum[idx3] += a_val * s3;
                }
            }
        }

        // Store the final results back to global memory
        for (size_t j = 0; j < K; ++j) {
            C_row[j] = accum[j];
        }
    }
}