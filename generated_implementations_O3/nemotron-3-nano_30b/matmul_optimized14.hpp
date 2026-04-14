#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5; // K / 32

    for (size_t i = 0; i < M; ++i) {
        // Temporary accumulator for this output row
        float accum[/*K*/];
        for (size_t j = 0; j < K; ++j) accum[j] = 0.0f;

        const float* A_row = A + i * K;
        float*       C_row = C + i * K;

        for (size_t p = 0; p < K; ++p) {
            float a_val = A_row[p];
            const uint32_t* B_row = B + p * K_ints;

            for (size_t chunk = 0; chunk < K_ints; ++chunk) {
                uint32_t word = B_row[chunk];
                size_t col_base = chunk * 32;

                // Process four columns per iteration
                #pragma unroll 8
                for (int offset = 0; offset < 32; offset += 4) {
                    uint32_t nibble = (word >> offset) & 0xF;
                    // Process each of the four bits
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        size_t col_idx = col_base + offset + t;
                        float sign = -1.0f + 2.0f * ((nibble >> t) & 1u);
                        accum[col_idx] += a_val * sign;
                    }
                }
            }
        }

        // Store the results
        for (size_t j = 0; j < K; ++j) {
            C_row[j] = accum[j];
        }
    }
}