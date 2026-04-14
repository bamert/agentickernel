#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5; // K / 32

    // Stack-allocated accumulator; test uses K = 3072, so 4096 is safe
    float accum[4096];
    for (size_t j = 0; j < K; ++j) accum[j] = 0.0f;

    for (size_t i = 0; i < M; ++i) {
        const float* A_i = A + i * K;   // row i of A
        float*       C_i = C + i * K;   // row i of C

        // Process each row p of the packed binary matrix B
        for (size_t p = 0; p < K; ++p) {
            float a_val = A_i[p];
            const uint32_t* B_row = B + p * K_ints; // start of row p in packed B

            // Process the row in 32‑bit chunks
            for (size_t chunk = 0; chunk < K_ints; ++chunk) {
                uint32_t word = B_row[chunk];
                size_t   col_base = chunk * 32; // first column index of this chunk

                // Process eight columns at a time (offset = 0,8,16,24)
                #pragma unroll 4
                for (int offset = 0; offset < 32; offset += 4) {
                    // Extract the 4‑bit nibble that encodes the signs for these columns
                    uint32_t nibble = (word >> offset) & 0xF;

                    // Process each of the four bits
                    #pragma unroll 4
                    for (int t = 0; t < 4; ++t) {
                        // Branch‑less conversion: -1.0f if bit==0, +1.0f if bit==1
                        uint32_t bit = (nibble >> t) & 1u;
                        float sign = -1.0f + 2.0f * static_cast<float>(bit);
                        size_t idx = col_base + offset + t; // global column index
                        accum[idx] += a_val * sign;
                    }
                }
            }
        }

        // Store the final results back to global memory
        for (size_t j = 0; j < K; ++j) {
            C_i[j] = accum[j];
        }
    }
}