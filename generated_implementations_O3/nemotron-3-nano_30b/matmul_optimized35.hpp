#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;   // K / 32

    // nibble -> four signs (+1 or -1)
    const float signTable[16][4] = {
        {-1,-1,-1,-1}, {+1,-1,-1,-1}, {-1,+1,-1,-1}, {+1,+1,-1,-1},
        {-1,-1,+1,-1}, {+1,-1,+1,-1}, {-1,+1,+1,-1}, {+1,+1,+1,-1},
        {-1,-1,-1,+1}, {+1,-1,-1,+1}, {-1,+1,-1,+1}, {+1,+1,-1,+1},
        {-1,-1,+1,+1}, {+1,-1,+1,+1}, {-1,+1,+1,+1}, {+1,+1,+1,+1}
    };

    // Process each output row i
    for (size_t i = 0; i < M; ++i) {
        // Zero the i‑th row of C before we start accumulating
        float* C_i = C + i * K;
        for (size_t j = 0; j < K; ++j) C_i[j] = 0.0f;

        const float* A_i = A + i * K;   // row i of A

        // Process each row p of the packed binary matrix B
        for (size_t p = 0; p < K; ++p) {
            float a_val = A_i[p];
            const uint32_t* B_row = B + p * K_ints;   // start of row p in packed B

            // Process the row in 32‑bit chunks
            for (size_t chunk = 0; chunk < K_ints; ++chunk) {
                uint32_t word = B_row[chunk];
                size_t   col_base = chunk * 32;        // first column index of this chunk

                // Process four columns at a time (offset = 0,4,8,...,28)
                #pragma unroll 8
                for (int offset = 0; offset < 32; offset += 4) {
                    // Extract the 4‑bit nibble that encodes the signs for these columns
                    uint32_t nibble = (word >> offset) & 0xF;

                    // Update the four consecutive elements of C_i
                    size_t idx0 = col_base + offset + 0;
                    size_t idx1 = col_base + offset + 1;
                    size_t idx2 = col_base + offset + 2;
                    size_t idx3 = col_base + offset + 3;

                    // Branch‑less sign extraction from the pre‑computed table
                    float s0 = signTable[nibble][0];
                    float s1 = signTable[nibble][1];
                    float s2 = signTable[nibble][2];
                    float s3 = signTable[nibble][3];

                    // Accumulate the contribution of a_val and the four signs
                    C_i[idx0] += a_val * s0;
                    C_i[idx1] += a_val * s1;
                    C_i[idx2] += a_val * s2;
                    C_i[idx3] += a_val * s3;
                }
            }
        }
    }
}