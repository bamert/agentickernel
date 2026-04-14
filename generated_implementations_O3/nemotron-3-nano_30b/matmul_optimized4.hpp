#pragma once
#include <cstdint>
#include <cstddef>

// Optimized matmul: process 4 columns at a time, reducing store traffic.
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K >> 5;  // K / 32

    for (size_t i = 0; i < M; ++i) {         // Each row of A (and C)
        const float* A_row = A + i * K;
        float*       C_row = C + i * K;

        // Process each row of packed B
        for (size_t p = 0; p < K; ++p) {
            float a_val = A_row[p];
            const uint32_t* B_row = B + p * K_ints;

            // Process 32‑bit chunks
            for (size_t chunk = 0; chunk < K_ints; ++chunk) {
                uint32_t word = B_row[chunk];

                // Process columns 4 at a time
                #pragma unroll 8
                for (size_t t = 0; t < 32; t += 4) {
                    // Load 4 consecutive elements of C_row
                    float vals[4] = {
                        C_row[chunk*32 + t + 0],
                        C_row[chunk*32 + t + 1],
                        C_row[chunk*32 + t + 2],
                        C_row[chunk*32 + t + 3]
                    };

                    // Update each with contribution of current a_val and sign bits
                    #pragma unroll
                    for (int offset = 0; offset < 4; ++offset) {
                        uint32_t bit = (word >> (t + offset)) & 1u;
                        float sign = bit ? 1.0f : -1.0f;
                        vals[offset] += a_val * sign;
                    }

                    // Store back the updated values
                    C_row[chunk*32 + t + 0] = vals[0];
                    C_row[chunk*32 + t + 1] = vals[1];
                    C_row[chunk*32 + t + 2] = vals[2];
                    C_row[chunk*32 + t + 3] = vals[3];
                }
            }
        }
    }
}