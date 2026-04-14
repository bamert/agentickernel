#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;               // K / 32

    // Map each 4‑bit pattern to four signs (+1 or –1)
    const float signTable[16][4] = {
        {-1,-1,-1,-1}, {+1,-1,-1,-1}, {-1,+1,-1,-1}, {+1,+1,-1,-1},
        {-1,-1,+1,-1}, {+1,-1,+1,-1}, {-1,+1,+1,-1}, {+1,+1,+1,-1},
        {-1,-1,-1,+1}, {+1,-1,-1,+1}, {-1,+1,-1,+1}, {+1,+1,-1,+1},
        {-1,-1,+1,+1}, {+1,-1,+1,+1}, {-1,+1,+1,+1}, {+1,+1,+1,+1}
    };

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

                // Process eight columns at a time (offset = 0,8,16,24)
                #pragma unroll 4
                for (int offset = 0; offset < 32; offset += 8) {
                    // Extract two nibbles that hold the signs for eight columns
                    uint32_t nibble0 = (word >> offset)       & 0xF;
                    uint32_t nibble1 = (word >> (offset + 4)) & 0xF;

                    const float* signs0 = signTable[nibble0];
                    const float* signs1 = signTable[nibble1];

                    // Column indices for the eight elements we will update
                    size_t idx0 = col_base + offset + 0;
                    size_t idx1 = col_base + offset + 1;
                    size_t idx2 = col_base + offset + 2;
                    size_t idx3 = col_base + offset + 3;
                    size_t idx4 = col_base + offset + 4;
                    size_t idx5 = col_base + offset + 5;
                    size_t idx6 = col_base + offset + 6;
                    size_t idx7 = col_base + offset + 7;

                    // Accumulate contributions of a_val and the eight signs
                    #pragma unroll 4
                    for (int t = 0; t < 4; ++t) {
                        accum[idx0 + t] += a_val * signs0[t];
                        accum[idx4 + t] += a_val * signs1[t];
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