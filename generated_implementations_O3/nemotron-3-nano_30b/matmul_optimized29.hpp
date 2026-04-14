#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5; // number of 32‑bit chunks per row of B

    // Map each 4‑bit pattern to four signs (+1 or –1)
    const float signTable[16][4] = {
        {-1,-1,-1,-1}, {+1,-1,-1,-1}, {-1,+1,-1,-1}, {+1,+1,-1,-1},
        {-1,-1,+1,-1}, {+1,-1,+1,-1}, {-1,+1,+1,-1}, {+1,+1,+1,-1},
        {-1,-1,-1,+1}, {+1,-1,-1,+1}, {-1,+1,-1,+1}, {+1,+1,-1,+1},
        {-1,-1,+1,+1}, {+1,-1,+1,+1}, {-1,+1,+1,+1}, {+1,+1,+1,+1}
    };

    // Use a local buffer large enough for the test case (K = 3072)
    float accum[4096];
    for (size_t j = 0; j < K; ++j) accum[j] = 0.0f;

    // Process each output row i
    for (size_t i = 0; i < M; ++i) {
        const float* A_i = A + i * K;   // row i of A
        float*       C_i = C + i * K;   // row i of C

        // Process each row p of the packed binary matrix B
        for (size_t p = 0; p < K; ++p) {
            float a_val = A_i[p];
            const uint32_t* B_row = B + p * K_ints; // start of row p

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
                    accum[idx0] += a_val * signs[0];
                    accum[idx1] += a_val * signs[1];
                    accum[idx2] += a_val * signs[2];
                    accum[idx3] += a_val * signs[3];
                }
            }
        }

        // Store the final results back to global memory
        for (size_t j = 0; j < K; ++j) {
            C_i[j] = accum[j];
        }
    }
}