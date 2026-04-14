#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5; // K / 32

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
                for (int offset = 0; offset < 32; offset += 8) {
                    // Extract two nibbles that hold the signs for eight columns
                    uint32_t nibble0 = (word >> offset)       & 0xF;
                    uint32_t nibble1 = (word >> (offset + 4)) & 0xF;

                    // Process each of the 4 bits in each nibble
                    #pragma unroll 4
                    for (int t = 0; t < 4; ++t) {
                        // Compute sign for bit t of nibble0 and nibble1 (branchless: -1 or +1)
                        float sign0 = -1.0f + 2.0f * static_cast<float>((nibble0 >> t) & 1u);
                        float sign1 = -1.0f + 2.0f * static_cast<float>((nibble1 >> t) & 1u);

                        // Column indices for the eight elements we will update
                        size_t idx0 = col_base + offset + t;
                        size_t idx1 = col_base + offset + t + 4;

                        // Accumulate contribution of a_val and the two signs
                        C_i[idx0] += a_val * sign0;
                        C_i[idx1] += a_val * sign1;
                    }
                }
            }
        }
    }
}