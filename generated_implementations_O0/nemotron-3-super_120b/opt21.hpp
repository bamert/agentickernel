#pragma once
#include <cstdint>
#include <cstddev>

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
            float a16 = A[i * K + p_start + 16];
            float a17 = A[i * K + p_start + 17];
            float a18 = A[i * K + p_start + 18];
            float a19 = A[i * K + p_start + 19];
            float a20 = A[i * K + p_start + 20];
            float a21 = A[i * K + p_start + 21];
            float a22 = A[i * K + p_start + 22];
            float a23 = A[i * K + p_start + 23];
            float a24 = A[i * K + p_start + 24];
            float a25 = A[i * K + p_start + 25];
            float a26 = A[i * K + p_start + 26];
            float a27 = A[i * K + p_start + 27];
            float a28 = A[i * K + p_start + 28];
            float a29 = A[i * K + p_start + 29];
            float a30 = A[i * K + p_start + 30];
            float a31 = A[i * K + p_start + 31];
            rsum += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 +
                    a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 +
                    a16 + a17 + a18 + a19 + a20 + a21 + a22 + a23 +
                    a24 + a25 + a26 + a27 + a28 + a29 + a30 + a31;

            // Array of a_vals for easier indexing
            float a_vals[32] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
                                a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31};

            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                size_t base_j = chunk_idx * 32;
                // Load 32 packed values (one for each of the 32 rows, for this chunk)
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
                uint32_t p16 = B[(p_start + 16) * K_ints + chunk_idx];
                uint32_t p17 = B[(p_start + 17) * K_ints + chunk_idx];
                uint32_t p18 = B[(p_start + 18) * K_ints + chunk_idx];
                uint32_t p19 = B[(p_start + 19) * K_ints + chunk_idx];
                uint32_t p20 = B[(p_start + 20) * K_ints + chunk_idx];
                uint32_t p21 = B[(p_start + 21) * K_ints + chunk_idx];
                uint32_t p22 = B[(p_start + 22) * K_ints + chunk_idx];
                uint32_t p23 = B[(p_start + 23) * K_ints + chunk_idx];
                uint32_t p24 = B[(p_start + 24) * K_ints + chunk_idx];
                uint32_t p25 = B[(p_start + 25) * K_ints + chunk_idx];
                uint32_t p26 = B[(p_start + 26) * K_ints + chunk_idx];
                uint32_t p27 = B[(p_start + 27) * K_ints + chunk_idx];
                uint32_t p28 = B[(p_start + 28) * K_ints + chunk_idx];
                uint32_t p29 = B[(p_start + 29) * K_ints + chunk_idx];
                uint32_t p30 = B[(p_start + 30) * K_ints + chunk_idx];
                uint32_t p31 = B[(p_start + 31) * K_ints + chunk_idx];
                // Array of packed values for easier indexing
                uint32_t packed_vals[32] = {p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
                                            p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30, p31};
                // Process each bit in the chunk (0..31) unrolled by 8
                for (size_t b = 0; b < 32; b += 8) {
                    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
                    float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
                    for (size_t ii = 0; ii < 32; ++ii) {
                        uint32_t packed = packed_vals[ii];
                        uint32_t bits8 = (packed >> b) & 0xFF;
                        if (bits8 & 0x01) sum0 += a_vals[ii];
                        if (bits8 & 0x02) sum1 += a_vals[ii];
                        if (bits8 & 0x04) sum2 += a_vals[ii];
                        if (bits8 & 0x08) sum3 += a_vals[ii];
                        if (bits8 & 0x10) sum4 += a_vals[ii];
                        if (bits8 & 0x20) sum5 += a_vals[ii];
                        if (bits8 & 0x40) sum6 += a_vals[ii];
                        if (bits8 & 0x80) sum7 += a_vals[ii];
                    }
                    C[i * K + base_j + b+0] += sum0;
                    C[i * K + base_j + b+1] += sum1;
                    C[i * K + base_j + b+2] += sum2;
                    C[i * K + base_j + b+3] += sum3;
                    C[i * K + base_j + b+4] += sum4;
                    C[i * K + base_j + b+5] += sum5;
                    C[i * K + base_j + b+6] += sum6;
                    C[i * K + base_j + b+7] += sum7;
                }
            }
        }
        // Final conversion: C[i,j] = 2.0f * C2[i,j] - row_sum[i]
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 2.0f * C[i * K + j] - rsum;
        }
    }
}