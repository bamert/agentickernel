#pragma once
#include <cstdint>
#include <cstddef>

// Optimized matmul – process rows in groups of 8.
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;               // number of 32‑bit chunks per row of B
    const size_t GROUP = 8;                     // process 8 rows at a time (M = 32 in the benchmark)

    // Temporary accumulator storage for the current group of rows
    // Assume K ≤ 4096 for the test case
    float group_acc[GROUP * 4096] = {0};

    for (size_t g = 0; g < M; g += GROUP) {
        // Process each row in the current group
        #pragma unroll 8
        for (size_t ii = 0; ii < GROUP && (g + ii) < M; ++ii) {
            size_t i   = g + ii;                                 // global row index
            float*   acc_i = &group_acc[ii * K];                 // pointer to this row’s accumulator
            const float* A_i = A + i * K;                        // row i of A
            float*       C_i = C + i * K;                        // row i of C

            // Process each row p of the packed binary matrix B
            for (size_t p = 0; p < K; ++p) {
                float a_val = A_i[p];
                const uint32_t* B_row = B + p * K_ints;            // start of row p in packed B

                // Process the row in 32‑bit chunks
                for (size_t chunk = 0; chunk < K_ints; ++chunk) {
                    uint32_t word = B_row[chunk];
                    size_t   col_base = chunk * 32;                 // first column index of this chunk

                    // Process eight columns at a time (offset = 0,8,16,24)
                    #pragma unroll 4
                    for (int offset = 0; offset < 32; offset += 8) {
                        // Extract the two nibbles that hold the signs for eight columns
                        uint32_t nibble0 = (word >> offset)       & 0xF;
                        uint32_t nibble1 = (word >> (offset + 4)) & 0xF;

                        // Process each of the four bits in nibble0
                        #pragma unroll 4
                        for (int t = 0; t < 4; ++t) {
                            // Branch‑less conversion: -1.0f if bit==0, +1.0f if bit==1
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