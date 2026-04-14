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
        // Process p in chunks of 16 (K is multiple of 32, hence multiple of 16)
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

            // Store a_vals in an array for reuse
            float a_vals[16] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15};

            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                size_t base_j = chunk_idx * 32;
                // Load 16 packed values (one for each of the 16 p's, for this chunk)
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
                // Process each bit in the chunk (0..31) in steps of 4
                for (size_t b = 0; b < 32; b += 4) {
                    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                    // ii = 0
                    uint32_t bits4 = (p0 >> b) & 0xF;
                    s0 += a0 * ((bits4 >> 0) & 1);
                    s1 += a0 * ((bits4 >> 1) & 1);
                    s2 += a0 * ((bits4 >> 2) & 1);
                    s3 += a0 * ((bits4 >> 3) & 1);
                    // ii = 1
                    bits4 = (p1 >> b) & 0xF;
                    s0 += a1 * ((bits4 >> 0) & 1);
                    s1 += a1 * ((bits4 >> 1) & 1);
                    s2 += a1 * ((bits4 >> 2) & 1);
                    s3 += a1 * ((bits4 >> 3) & 1);
                    // ii = 2
                    bits4 = (p2 >> b) & 0xF;
                    s0 += a2 * ((bits4 >> 0) & 1);
                    s1 += a2 * ((bits4 >> 1) & 1);
                    s2 += a2 * ((bits4 >> 2) & 1);
                    s3 += a2 * ((bits4 >> 3) & 1);
                    // ii = 3
                    bits4 = (p3 >> b) & 0xF;
                    s0 += a3 * ((bits4 >> 0) & 1);
                    s1 += a3 * ((bits4 >> 1) & 1);
                    s2 += a3 * ((bits4 >> 2) & 1);
                    s3 += a3 * ((bits4 >> 3) & 1);
                    // ii = 4
                    bits4 = (p4 >> b) & 0xF;
                    s0 += a4 * ((bits4 >> 0) & 1);
                    s1 += a4 * ((bits4 >> 1) & 1);
                    s2 += a4 * ((bits4 >> 2) & 1);
                    s3 += a4 * ((bits4 >> 3) & 1);
                    // ii = 5
                    bits4 = (p5 >> b) & 0xF;
                    s0 += a5 * ((bits4 >> 0) & 1);
                    s1 += a5 * ((bits4 >> 1) & 1);
                    s2 += a5 * ((bits4 >> 2) & 1);
                    s3 += a5 * ((bits4 >> 3) & 1);
                    // ii = 6
                    bits4 = (p6 >> b) & 0xF;
                    s0 += a6 * ((bits4 >> 0) & 1);
                    s1 += a6 * ((bits4 >> 1) & 1);
                    s2 += a6 * ((bits4 >> 2) & 1);
                    s3 += a6 * ((bits4 >> 3) & 1);
                    // ii = 7
                    bits4 = (p7 >> b) & 0xF;
                    s0 += a7 * ((bits4 >> 0) & 1);
                    s1 += a7 * ((bits4 >> 1) & 1);
                    s2 += a7 * ((bits4 >> 2) & 1);
                    s3 += a7 * ((bits4 >> 3) & 1);
                    // ii = 8
                    bits4 = (p8 >> b) & 0xF;
                    s0 += a8 * ((bits4 >> 0) & 1);
                    s1 += a8 * ((bits4 >> 1) & 1);
                    s2 += a8 * ((bits4 >> 2) & 1);
                    s3 += a8 * ((bits4 >> 3) & 1);
                    // ii = 9
                    bits4 = (p9 >> b) & 0xF;
                    s0 += a9 * ((bits4 >> 0) & 1);
                    s1 += a9 * ((bits4 >> 1) & 1);
                    s2 += a9 * ((bits4 >> 2) & 1);
                    s3 += a9 * ((bits4 >> 3) & 1);
                    // ii = 10
                    bits4 = (p10 >> b) & 0xF;
                    s0 += a10 * ((bits4 >> 0) & 1);
                    s1 += a10 * ((bits4 >> 1) & 1);
                    s2 += a10 * ((bits4 >> 2) & 1);
                    s3 += a10 * ((bits4 >> 3) & 1);
                    // ii = 11
                    bits4 = (p11 >> b) & 0xF;
                    s0 += a11 * ((bits4 >> 0) & 1);
                    s1 += a11 * ((bits4 >> 1) & 1);
                    s2 += a11 * ((bits4 >> 2) & 1);
                    s3 += a11 * ((bits4 >> 3) & 1);
                    // ii = 12
                    bits4 = (p12 >> b) & 0xF;
                    s0 += a12 * ((bits4 >> 0) & 1);
                    s1 += a12 * ((bits4 >> 1) & 1);
                    s2 += a12 * ((bits4 >> 2) & 1);
                    s3 += a12 * ((bits4 >> 3) & 1);
                    // ii = 13
                    bits4 = (p13 >> b) & 0xF;
                    s0 += a13 * ((bits4 >> 0) & 1);
                    s1 += a13 * ((bits4 >> 1) & 1);
                    s2 += a13 * ((bits4 >> 2) & 1);
                    s3 += a13 * ((bits4 >> 3) & 1);
                    // ii = 14
                    bits4 = (p14 >> b) & 0xF;
                    s0 += a14 * ((bits4 >> 0) & 1);
                    s1 += a14 * ((bits4 >> 1) & 1);
                    s2 += a14 * ((bits4 >> 2) & 1);
                    s3 += a14 * ((bits4 >> 3) & 1);
                    // ii = 15
                    bits4 = (p15 >> b) & 0xF;
                    s0 += a15 * ((bits4 >> 0) & 1);
                    s1 += a15 * ((bits4 >> 1) & 1);
                    s2 += a15 * ((bits4 >> 2) & 1);
                    s3 += a15 * ((bits4 >> 3) & 1);
                    // Store the 4 sums
                    C[i * K + base_j + b + 0] += s0;
                    C[i * K + base_j + b + 1] += s1;
                    C[i * K + base_j + b + 2] += s2;
                    C[i * K + base_j + b + 3] += s3;
                }
            }
        }
        // Final conversion: C[i,j] = 2.0f * C2[i,j] - row_sum[i]
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 2.0f * C[i * K + j] - rsum;
        }
    }
}