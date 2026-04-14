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

    // Process rows of A in groups of 8 (M = 32 in the benchmark)
    const size_t GROUP = 8;
    // Accumulator storage: one block per row in the group (K ≤ 4096 for the test)
    float group_acc[GROUP * 4096] = {0};

    for (size_t g = 0; g < M; g += GROUP) {
        // Process each row in the current group
        #pragma unroll
        for (size_t ii = 0; ii < GROUP && (g + ii) < M; ++ii) {
            size_t i   = g + ii;                                    // global row index
            float*   acc_i = &group_acc[ii * K];                    // pointer to this row’s accumulator
            const float* A_i = A + i * K;                           // row i of A
            float*       C_i = C + i * K;                           // row i of C

            // Process each row p of the packed binary matrix B
            for (size_t p = 0; p < K; ++p) {
                float a_val = A_i[p];
                const uint32_t* B_row = B + p * K_ints;               // start of row p in packed B

                // Process the row in 32‑bit chunks
                for (size_t chunk = 0; chunk < K_ints; ++chunk) {
                    uint32_t word = B_row[chunk];
                    size_t   col_base = chunk * 32;                     // first column index of this chunk

                    // Process eight columns at a time (offset = 0,8,16,24)
                    #pragma unroll 4
                    for (int offset = 0; offset < 32; offset += 4) {
                        // Extract two nibbles that hold the signs for eight columns
                        uint32_t nibble0 = (word >> offset)       & 0xF;
                        uint32_t nibble1 = (word >> (offset + 4)) & 0xF;

                        // Process each of the four bits in nibble0
                        #pragma unroll 4
                        for (int t = 0; t < 4; ++t) {
                            float sign = -1.0f + 2.0f * static_cast<float>((nibble0 >> t) & 1u);
                            size_t idx = col_base + offset + t;           // global column index
                            acc_i[idx] += a_val * sign;
                        }
                        // Process each of the four bits in nibble1
                        #pragma unroll 4
                        for (int t = 0; t < 4; ++t) {
                            float sign = -1.0f + 2.0f * static_cast<float>((nibble1 >> t) & 1u);
                            size_t idx = col_base + offset + t + 4;       // global column index
                            acc_i[idx] += a_val * sign;
                        }
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