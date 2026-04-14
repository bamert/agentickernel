#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        // Initialize the row of C to zero (for C2 accumulation)
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }
        float rsum = 0.0f;
        // Process p in chunks of 32 (K is multiple of 32)
        for (size_t p_start = 0; p_start < K; p_start += 32) {
            // Load 32 a_vals
            float a_vals[32];
            for (size_t ii = 0; ii < 32; ++ii) {
                a_vals[ii] = A[i * K + p_start + ii];
                rsum += a_vals[ii];
            }

            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                // Load 32 packed values (one for each of the 32 rows, for this chunk)
                uint32_t packed_vals[32];
                for (size_t ii = 0; ii < 32; ++ii) {
                    packed_vals[ii] = B[(p_start + ii) * K_ints + chunk_idx];
                }
                size_t base_j = chunk_idx * 32;
                // Process each bit in the chunk (0..31)
                for (size_t b = 0; b < 32; ++b) {
                    float sum = 0.0f;
                    for (size_t ii = 0; ii < 32; ++ii) {
                        uint32_t bit = (packed_vals[ii] >> b) & 1;
                        sum += a_vals[ii] * bit;
                    }
                    C[i * K + base_j + b] += sum;
                }
            }
        }
        // Final conversion: C[i,j] = 2.0f * C2[i,j] - row_sum[i]
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 2.0f * C[i * K + j] - rsum;
        }
    }
}