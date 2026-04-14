#pragma once
#include <cstdint>
#include <cstddef>

// Optimized matmul using NEON‑style vector operations (4 lanes at a time)
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;  // K / 32

    // Process each row of A and C
    for (size_t i = 0; i < M; ++i) {
        const float* A_row = A + i * K;   // Row i of A
        float*       C_row = C + i * K;   // Row i of C

        // Process the B matrix in 32‑bit chunks
        for (size_t chunk = 0; chunk < K_ints; ++chunk) {
            size_t col_base = chunk * 32;  // First column index of this chunk

            // Process four columns at a time (offset 0,4,8,...,28)
            #pragma unroll 8
            for (int offset = 0; offset < 32; offset += 4) {
                // Global column indices for these four elements
                size_t col_idx0 = col_base + offset + 0;
                size_t col_idx1 = col_base + offset + 1;
                size_t col_idx2 = col_base + offset + 2;
                size_t col_idx3 = col_base + offset + 3;

                // Vector accumulator for the four columns
                float32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

                // Accumulate contributions from all rows p of B
                for (size_t p = 0; p < K; ++p) {
                    float a_val = A_row[p];
                    const uint32_t* B_word = B + p * K_ints + chunk;
                    uint32_t word = *B_word;

                    // Extract the four bits we care about
                    uint32_t bits = (word >> offset) & 0xF;   // bits 0‑3 in this group

                    // Convert each bit to +1.0f or -1.0f (branchless)
                    float s0 = ((bits >> 0) & 1u) ? 1.0f : -1.0f;
                    float s1 = ((bits >> 1) & 1u) ? 1.0f : -1.0f;
                    float s2 = ((bits >> 2) & 1u) ? 1.0f : -1.0f;
                    float s3 = ((bits >> 3) & 1u) ? 1.0f : -1.0f;

                    // Pack the four signs into a vector
                    float32x4_t sign_vec = (float32x4_t){s0, s1, s2, s3};

                    // Scale by a_val and add to the accumulator
                    float32x4_t term = vmulq_n_f32(sign_vec, a_val);
                    acc = vaddq_f32(acc, term);
                }

                // Store the accumulated results back to C
                C_row[col_idx0] += vgetq_lane_f32(acc, 0);
                C_row[col_idx1] += vgetq_lane_f32(acc, 1);
                C_row[col_idx2] += vgetq_lane_f32(acc, 2);
                C_row[col_idx3] += vgetq_lane_f32(acc, 3);
            }
        }
    }
}