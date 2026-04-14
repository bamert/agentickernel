#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5; // K / 32

    for (size_t i = 0; i < M; ++i) {
        const float* A_i = A + i * K;
        float*       C_i = C + i * K;

        // Process each 32‑bit chunk of B
        #pragma unroll 4
        for (size_t chunk = 0; chunk < K_ints; ++chunk) {
            size_t col_base = chunk * 32;

            // Process four bits at a time
            #pragma unroll 8
            for (int offset = 0; offset < 32; offset += 4) {
                // Column indices for the 4 elements we will update
                size_t idx0 = col_base + offset + 0;
                size_t idx1 = col_base + offset + 1;
                size_t idx2 = col_base + offset + 2;
                size_t idx3 = col_base + offset + 3;

                // Accumulators for these four columns
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;

                // Process all rows p of the packed matrix
                #pragma unroll 4
                for (size_t p = 0; p < K; ++p) {
                    float a_val = A_i[p];
                    const uint32_t* B_ptr = B + p * K_ints + chunk;
                    uint32_t word = *B_ptr;

                    // Extract the 4 bits starting at 'offset'
                    uint32_t nibble = (word >> offset) & 0xF;

                    // Convert bits to +1.0f or -1.0f without branching
                    float s0 = -1.0f + 2.0f * ((nibble >> 0) & 1u);
                    float s1 = -1.0f + 2.0f * ((nibble >> 1) & 1u);
                    float s2 = -1.0f + 2.0f * ((nibble >> 2) & 1u);
                    float s3 = -1.0f + 2.0f * ((nibble >> 3) & 1u);

                    // Accumulate contribution
                    acc0 += a_val * s0;
                    acc1 += a_val * s1;
                    acc2 += a_val * s2;
                    acc3 += a_val * s3;
                }

                // Store final results
                C_i[idx0] = acc0;
                C_i[idx1] = acc1;
                C_i[idx2] = acc2;
                C_i[idx3] = acc3;
            }
        }
    }
}