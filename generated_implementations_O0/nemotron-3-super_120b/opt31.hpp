#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        // Initialize the row of C to zero (for C2)
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }
        float rsum = 0.0f;
        size_t p = 0;
        // Process p in chunks of 16 (K is multiple of 32, hence multiple of 16)
        for (; p < K; p += 16) {
            // Load 16 a_vals and accumulate rsum
            float a0  = A[i * K + p + 0];
            float a1  = A[i * K + p + 1];
            float a2  = A[i * K + p + 2];
            float a3  = A[i * K + p + 3];
            float a4  = A[i * K + p + 4];
            float a5  = A[i * K + p + 5];
            float a6  = A[i * K + p + 6];
            float a7  = A[i * K + p + 7];
            float a8  = A[i * K + p + 8];
            float a9  = A[i * K + p + 9];
            float a10 = A[i * K + p + 10];
            float a11 = A[i * K + p + 11];
            float a12 = A[i * K + p + 12];
            float a13 = A[i * K + p + 13];
            float a14 = A[i * K + p + 14];
            float a15 = A[i * K + p + 15];
            rsum += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;

            // Store a_vals in an array for reuse
            float a_vals[16] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15};

            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                size_t base_j = chunk_idx * 32;
                // Load 16 packed values for this chunk (one for each of the 16 p's)
                uint32_t packed[16];
                for (size_t ii = 0; ii < 16; ++ii) {
                    packed[ii] = B[(p + ii) * K_ints + chunk_idx];
                }
                // Process each of the 16 p's
                for (size_t ii = 0; ii < 16; ++ii) {
                    uint32_t packed_val = packed[ii];
                    float a_val = a_vals[ii];
                    // Process 4 bits at a time
                    for (size_t b = 0; b < 32; b += 4) {
                        uint32_t bits4 = (packed_val >> b) & 0xF;
                        C[i * K + base_j + b + 0] += a_val * ((bits4 >> 0) & 1);
                        C[i * K + base_j + b + 1] += a_val * ((bits4 >> 1) & 1);
                        C[i * K + base_j + b + 2] += a_val * ((bits4 >> 2) & 1);
                        C[i * K + base_j + b + 3] += a_val * ((bits4 >> 3) & 1);
                    }
                }
            }
        }
        // Final conversion: C[i,j] = 2.0f * C2[i,j] - row_sum[i]
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 2.0f * C[i * K + j] - rsum;
        }
    }
}