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
        // Process p in chunks of 16
        for (size_t p_start = 0; p_start < K; p_start += 16) {
            // Load 16 a_vals
            float a0  = A[i * K + p_start + 0];
            float a1  = A[i * K + p_start + 1];
            float a2  = A[i * K + p_start + 2];
            float a3  = A[i * K + p_start + 3];
            float a4  = A[i * K + p_start + 4];
            float a5  = A[i * K + p_start + 5];
            float a6  = A[i * K + p_start + 6];
            float a7  = A[i * K + p_start + 7];
            float a8  = A[i * K + p_start + 8];
            float a9  = A[i * K + p_start + 9];
            float a10 = A[i * K + p_start + 10];
            float a11 = A[i * K + p_start + 11];
            float a12 = A[i * K + p_start + 12];
            float a13 = A[i * K + p_start + 13];
            float a14 = A[i * K + p_start + 14];
            float a15 = A[i * K + p_start + 15];
            rsum += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 +
                    a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;

            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                size_t base_j = chunk_idx * 32;
                // Load 16 packed values (one for each of the 16 rows, for this chunk)
                uint32_t p0  = B[(p_start + 0) * K_ints + chunk_idx];
                uint32_t p1  = B[(p_start + 1) * K_ints + chunk_idx];
                uint32_t p2  = B[(p_start + 2) * K_ints + chunk_idx];
                uint32_t p3  = B[(p_start + 3) * K_ints + chunk_idx];
                uint32_t p4  = B[(p_start + 4) * K_ints + chunk_idx];
                uint32_t p5  = B[(p_start + 5) * K_ints + chunk_idx];
                uint32_t p6  = B[(p_start + 6) * K_ints + chunk_idx];
                uint32_t p7  = B[(p_start + 7) * K_ints + chunk_idx];
                uint32_t p8  = B[(p_start + 8) * K_ints + chunk_idx];
                uint32_t p9  = B[(p_start + 9) * K_ints + chunk_idx];
                uint32_t p10 = B[(p_start + 10) * K_ints + chunk_idx];
                uint32_t p11 = B[(p_start + 11) * K_ints + chunk_idx];
                uint32_t p12 = B[(p_start + 12) * K_ints + chunk_idx];
                uint32_t p13 = B[(p_start + 13) * K_ints + chunk_idx];
                uint32_t p14 = B[(p_start + 14) * K_ints + chunk_idx];
                uint32_t p15 = B[(p_start + 15) * K_ints + chunk_idx];
                // Process each bit in the chunk (0..31) with inner loop over ii unrolled by 4
                for (size_t b = 0; b < 32; ++b) {
                    float sum = 0.0f;
                    // ii = 0
                    uint32_t bit0 = (p0 >> b) & 1; sum += a0 * bit0;
                    // ii = 1
                    uint32_t bit1 = (p1 >> b) & 1; sum += a1 * bit1;
                    // ii = 2
                    uint32_t bit2 = (p2 >> b) & 1; sum += a2 * bit2;
                    // ii = 3
                    uint32_t bit3 = (p3 >> b) & 1; sum += a3 * bit3;
                    // ii = 4
                    uint32_t bit4 = (p4 >> b) & 1; sum += a4 * bit4;
                    // ii = 5
                    uint32_t bit5 = (p5 >> b) & 1; sum += a5 * bit5;
                    // ii = 6
                    uint32_t bit6 = (p6 >> b) & 1; sum += a6 * bit6;
                    // ii = 7
                    uint32_t bit7 = (p7 >> b) & 1; sum += a7 * bit7;
                    // ii = 8
                    uint32_t bit8 = (p8 >> b) & 1; sum += a8 * bit8;
                    // ii = 9
                    uint32_t bit9 = (p9 >> b) & 1; sum += a9 * bit9;
                    // ii = 10
                    uint32_t bit10 = (p10 >> b) & 1; sum += a10 * bit10;
                    // ii = 11
                    uint32_t bit11 = (p11 >> b) & 1; sum += a11 * bit11;
                    // ii = 12
                    uint32_t bit12 = (p12 >> b) & 1; sum += a12 * bit12;
                    // ii = 13
                    uint32_t bit13 = (p13 >> b) & 1; sum += a13 * bit13;
                    // ii = 14
                    uint32_t bit14 = (p14 >> b) & 1; sum += a14 * bit14;
                    // ii = 15
                    uint32_t bit15 = (p15 >> b) & 1; sum += a15 * bit15;
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