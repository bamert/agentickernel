#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5; // K / 32

    // nibble -> four signs (+1 or -1)
    const float signTable[16][4] = {
        {-1,-1,-1,-1}, {+1,-1,-1,-1}, {-1,+1,-1,-1}, {+1,+1,-1,-1},
        {-1,-1,+1,-1}, {+1,-1,+1,-1}, {-1,+1,+1,-1}, {+1,+1,+1,-1},
        {-1,-1,-1,+1}, {+1,-1,-1,+1}, {-1,+1,-1,+1}, {+1,+1,-1,+1},
        {-1,-1,+1,+1}, {+1,-1,+1,+1}, {-1,+1,+1,+1}, {+1,+1,+1,+1}
    };

    // Accumulator for each column of the output row
    float accum[4096]; // Enough for K up to 4096
    for (size_t j = 0; j < K; ++j) accum[j] = 0.0f;

    // Process each row p of the packed binary matrix B
    for (size_t p = 0; p < K; ++p) {
        // Load the scalar values A[i*K + p] for all output rows i
        float a_vals[32]; // M <= 32 in the benchmark
        #pragma unroll 4
        for (size_t i = 0; i < M; ++i) {
            a_vals[i] = A[i * K + p];
        }

        const uint32_t* B_row = B + p * K_ints; // start of row p

        // Process row in 32‑bit chunks
        for (size_t chunk = 0; chunk < K_ints; ++chunk) {
            uint32_t word = B_row[chunk];
            size_t col_base = chunk * 32; // first column index of this chunk

            // Process eight columns at a time (offset = 0,8,16,24)
            #pragma unroll 4
            for (int offset = 0; offset < 32; offset += 8) {
                // Extract two nibbles holding the signs for eight columns
                uint32_t nibble0 = (word >> offset)       & 0xF;
                uint32_t nibble1 = (word >> (offset + 4)) & 0xF;

                const float* signs0 = signTable[nibble0];
                const float* signs1 = signTable[nibble1];

                // Column indices for the eight elements
                size_t idx0 = col_base + offset + 0;
                size_t idx1 = col_base + offset + 1;
                size_t idx2 = col_base + offset + 2;
                size_t idx3 = col_base + offset + 3;
                size_t idx4 = col_base + offset + 4;
                size_t idx5 = col_base + offset + 5;
                size_t idx6 = col_base + offset + 6;
                size_t idx7 = col_base + offset + 7;

                // Update the accumulators for all output rows
                #pragma unroll 4
                for (size_t i = 0; i < M; ++i) {
                    accum[idx0] += a_vals[i] * signs0[0];
                    accum[idx1] += a_vals[i] * signs0[1];
                    accum[idx2] += a_vals[i] * signs0[2];
                    accum[idx3] += a_vals[i] * signs0[3];

                    accum[idx4] += a_vals[i] * signs1[0];
                    accum[idx5] += a_vals[i] * signs1[1];
                    accum[idx6] += a_vals[i] * signs1[2];
                    accum[idx7] += a_vals[i] * signs1[3];
                }
            }
        }
    }

    // Store the results back to global memory
    for (size_t i = 0; i < M; ++i) {
        float* C_row = C + i * K;
        #pragma unroll 4
        for (size_t j = 0; j < K; ++j) {
            C_row[j] = accum[j];
        }
    }
}