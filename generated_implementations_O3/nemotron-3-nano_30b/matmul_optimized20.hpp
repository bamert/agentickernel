#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;  // K / 32

    for (size_t i = 0; i < M; ++i) {
        // Accumulator array for this output row (K is known at compile‑time in the test)
        float accum[4096];  // Large enough for the test case (K = 3072)
        for (size_t j = 0; j < K; ++j) accum[j] = 0.0f;

        const float* A_row = A + i * K;   // row i of A
        float*       C_row = C + i * K;   // row i of C

        // Process each row p of the packed binary matrix B
        for (size_t p = 0; p < K; ++p) {
            float a_val = A_row[p];
            const uint32_t* B_row = B + p * K_ints;  // start of row p in packed B

            // Process the row in 32‑bit chunks
            for (size_t chunk = 0; chunk < K_ints; ++chunk) {
                uint32_t word = B_row[chunk];
                size_t   col_base = chunk * 32;        // first column index of this chunk

                // Process eight columns at a time
                #pragma unroll 8
                for (int offset = 0; offset < 32; offset += 8) {
                    // Update eight consecutive elements of the accumulator
                    #pragma unroll 8
                    for (int t = 0; t < 8; ++t) {
                        size_t idx = col_base + offset + t;          // global column index
                        uint32_t bit = (word >> (offset + t)) & 1u;   // extract bit t
                        float sign = (bit << 1) - 1.0f;              // -1.0f if bit==0, +1.0f if bit==1
                        accum[idx] += a_val * sign;                  // accumulate contribution
                    }
                }
            }
        }

        // Store the final results back to global memory
        for (size_t j = 0; j < K; ++j) {
            C_row[j] = accum[j];
        }
    }
}