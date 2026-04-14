#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;
    // nibble -> four signs (+1 or -1)
    const float signTable[16][4] = {
        {-1,-1,-1,-1}, {+1,-1,-1,-1}, {-1,+1,-1,-1}, {+1,+1,-1,-1},
        {-1,-1,+1,-1}, {+1,-1,+1,-1}, {-1,+1,+1,-1}, {+1,+1,+1,-1},
        {-1,-1,-1,+1}, {+1,-1,-1,+1}, {-1,+1,-1,+1}, {+1,+1,-1,+1},
        {-1,-1,+1,+1}, {+1,-1,+1,+1}, {-1,+1,+1,+1}, {+1,+1,+1,+1}
    };

    const size_t GROUP = 4; // process 4 rows at a time (M = 32 in benchmark)
    // Accumulator storage: one block per row in the group (K <= 4096)
    float acc[GROUP * 4096] = {}; // zero‑initialized

    // Process groups of rows
    for (size_t g = 0; g < M; g += GROUP) {
        // Process each row in the current group
        #pragma unroll 4
        for (size_t ii = 0; ii < GROUP && (g + ii) < M; ++ii) {
            size_t i = g + ii; // global row index
            float* acc_i = &acc[ii * K]; // pointer to this row’s accumulator
            const float* A_i = A + i * K; // row i of A
            float*       C_i = C + i * K; // row i of C

            // Process each row p of the packed binary matrix B
            for (size_t p = 0; p < K; ++p) {
                float a_val = A_i[p];
                const uint32_t* B_row = B + p * K_ints; // start of row p in packed B

                // Process the row in 32‑bit chunks
                for (size_t chunk = 0; chunk < K_ints; ++chunk) {
                    uint32_t word = B_row[chunk];
                    size_t   col_base = chunk * 32; // first column index of this chunk

                    // Process four columns per iteration
                    #pragma unroll 8
                    for (int offset = 0; offset < 32; offset += 4) {
                        // Extract the 4‑bit nibble that encodes the signs for these columns
                        uint32_t nibble = (word >> offset) & 0xF;
                        const float* signs = signTable[nibble]; // points to 4 pre‑computed signs

                        // Global column indices for the four elements we will update
                        size_t idx0 = col_base + offset + 0;
                        size_t idx1 = col_base + offset + 1;
                        size_t idx2 = col_base + offset + 2;
                        size_t idx3 = col_base + offset + 3;

                        // Accumulate the contribution of a_val and the four signs
                        acc_i[idx0] += a_val * signs[0];
                        acc_i[idx1] += a_val * signs[1];
                        acc_i[idx2] += a_val * signs[2];
                        acc_i[idx3] += a_val * signs[3];
                    }
                }
            }

            // Store the final results back to global memory
            #pragma unroll 4
            for (size_t j = 0; j < K; ++j) {
                C_i[j] = acc_i[j];
            }
        }
    }
}