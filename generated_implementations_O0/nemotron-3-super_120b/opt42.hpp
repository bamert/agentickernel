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
        size_t p = 0;
        // Process p in chunks of 32 (K is multiple of 32)
        for (; p <= K - 32; p += 32) {
            // Load 32 a_vals
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
            float a16 = A[i * K + p + 16];
            float a17 = A[i * K + p + 17];
            float a18 = A[i * K + p + 18];
            float a19 = A[i * K + p + 19];
            float a20 = A[i * K + p + 20];
            float a21 = A[i * K + p + 21];
            float a22 = A[i * K + p + 22];
            float a23 = A[i * K + p + 23];
            float a24 = A[i * K + p + 24];
            float a25 = A[i * K + p + 25];
            float a26 = A[i * K + p + 26];
            float a27 = A[i * K + p + 27];
            float a28 = A[i * K + p + 28];
            float a29 = A[i * K + p + 29];
            float a30 = A[i * K + p + 30];
            float a31 = A[i * K + p + 31];
            rsum += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 +
                    a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 +
                    a16 + a17 + a18 + a19 + a20 + a21 + a22 + a23 +
                    a24 + a25 + a26 + a27 + a28 + a29 + a30 + a31;

            // Store a_vals in an array for reuse
            float a_vals[32] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
                                a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31};

            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                size_t base_j = chunk_idx * 32;
                // Load 32 packed values (one for each of the 32 rows, for this chunk)
                uint32_t p0  = B[(p + 0) * K_ints + chunk_idx];
                uint32_t p1  = B[(p + 1) * K_ints + chunk_idx];
                uint32_t p2  = B[(p + 2) * K_ints + chunk_idx];
                uint32_t p3  = B[(p + 3) * K_ints + chunk_idx];
                uint32_t p4  = B[(p + 4) * K_ints + chunk_idx];
                uint32_t p5  = B[(p + 5) * K_ints + chunk_idx];
                uint32_t p6  = B[(p + 6) * K_ints + chunk_idx];
                uint32_t p7  = B[(p + 7) * K_ints + chunk_idx];
                uint32_t p8  = B[(p + 8) * K_ints + chunk_idx];
                uint32_t p9  = B[(p + 9) * K_ints + chunk_idx];
                uint32_t p10 = B[(p + 10) * K_ints + chunk_idx];
                uint32_t p11 = B[(p + 11) * K_ints + chunk_idx];
                uint32_t p12 = B[(p + 12) * K_ints + chunk_idx];
                uint32_t p13 = B[(p + 13) * K_ints + chunk_idx];
                uint32_t p14 = B[(p + 14) * K_ints + chunk_idx];
                uint32_t p15 = B[(p + 15) * K_ints + chunk_idx];
                uint32_t p16 = B[(p + 16) * K_ints + chunk_idx];
                uint32_t p17 = B[(p + 17) * K_ints + chunk_idx];
                uint32_t p18 = B[(p + 18) * K_ints + chunk_idx];
                uint32_t p19 = B[(p + 19) * K_ints + chunk_idx];
                uint32_t p20 = B[(p + 20) * K_ints + chunk_idx];
                uint32_t p21 = B[(p + 21) * K_ints + chunk_idx];
                uint32_t p22 = B[(p + 22) * K_ints + chunk_idx];
                uint32_t p23 = B[(p + 23) * K_ints + chunk_idx];
                uint32_t p24 = B[(p + 24) * K_ints + chunk_idx];
                uint32_t p25 = B[(p + 25) * K_ints + chunk_idx];
                uint32_t p26 = B[(p + 26) * K_ints + chunk_idx];
                uint32_t p27 = B[(p + 27) * K_ints + chunk_idx];
                uint32_t p28 = B[(p + 28) * K_ints + chunk_idx];
                uint32_t p29 = B[(p + 29) * K_ints + chunk_idx];
                uint32_t p30 = B[(p + 30) * K_ints + chunk_idx];
                uint32_t p31 = B[(p + 31) * K_ints + chunk_idx];
                // Process each bit in the chunk (0..31) with inner loop over ii unrolled by 4
                for (size_t b = 0; b < 32; ++b) {
                    float sum = 0.0f;
                    // ii = 0..3
                    sum += a_vals[0] * ((p0  >> b) & 1u);
                    sum += a_vals[1] * ((p1  >> b) & 1u);
                    sum += a_vals[2] * ((p2  >> b) & 1u);
                    sum += a_vals[3] * ((p3  >> b) & 1u);
                    // ii = 4..7
                    sum += a_vals[4] * ((p4  >> b) & 1u);
                    sum += a_vals[5] * ((p5  >> b) & 1u);
                    sum += a_vals[6] * ((p6  >> b) & 1u);
                    sum += a_vals[7] * ((p7  >> b) & 1u);
                    // ii = 8..11
                    sum += a_vals[8] * ((p8  >> b) & 1u);
                    sum += a_vals[9] * ((p9  >> b) & 1u);
                    sum += a_vals[10] * ((p10 >> b) & 1u);
                    sum += a_vals[11] * ((p11 >> b) & 1u);
                    // ii = 12..15
                    sum += a_vals[12] * ((p12 >> b) & 1u);
                    sum += a_vals[13] * ((p13 >> b) & 1u);
                    sum += a_vals[14] * ((p14 >> b) & 1u);
                    sum += a_vals[15] * ((p15 >> b) & 1u);
                    // ii = 16..19
                    sum += a_vals[16] * ((p16 >> b) & 1u);
                    sum += a_vals[17] * ((p17 >> b) & 1u);
                    sum += a_vals[18] * ((p18 >> b) & 1u);
                    sum += a_vals[19] * ((p19 >> b) & 1u);
                    // ii = 20..23
                    sum += a_vals[20] * ((p20 >> b) & 1u);
                    sum += a_vals[21] * ((p21 >> b) & 1u);
                    sum += a_vals[22] * ((p22 >> b) & 1u);
                    sum += a_vals[23] * ((p23 >> b) & 1u);
                    // ii = 24..27
                    sum += a_vals[24] * ((p24 >> b) & 1u);
                    sum += a_vals[25] * ((p25 >> b) & 1u);
                    sum += a_vals[26] * ((p26 >> b) & 1u);
                    sum += a_vals[27] * ((p27 >> b) & 1u);
                    // ii = 28..31
                    sum += a_vals[28] * ((p28 >> b) & 1u);
                    sum += a_vals[29] * ((p29 >> b) & 1u);
                    sum += a_vals[30] * ((p30 >> b) & 1u);
                    sum += a_vals[31] * ((p31 >> b) & 1u);
                    C[i * K + base_j + b] += sum;
                }
            }
        }
        // Remainder loop for p (should be zero because K is multiple of 32)
        for (; p < K; ++p) {
            float a_val = A[i * K + p];
            rsum += a_val;
            const uint32_t* B_row = B + p * K_ints;
            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                uint32_t packed = B_row[chunk_idx];
                size_t base_j = chunk_idx * 32;
                // Fully unroll the 32 bits
                uint32_t bits0  = (packed >> 0)  & 1; C[i * K + base_j + 0]  += a_val * bits0;
                uint32_t bits1  = (packed >> 1)  & 1; C[i * K + base_j + 1]  += a_val * bits1;
                uint32_t bits2  = (packed >> 2)  & 1; C[i * K + base_j + 2]  += a_val * bits2;
                uint32_t bits3  = (packed >> 3)  & 1; C[i * K + base_j + 3]  += a_val * bits3;
                uint32_t bits4  = (packed >> 4)  & 1; C[i * K + base_j + 4]  += a_val * bits4;
                uint32_t bits5  = (packed >> 5)  & 1; C[i * K + base_j + 5]  += a_val * bits5;
                uint32_t bits6  = (packed >> 6)  & 1; C[i * K + base_j + 6]  += a_val * bits6;
                uint32_t bits7  = (packed >> 7)  & 1; C[i * K + base_j + 7]  += a_val * bits7;
                uint32_t bits8  = (packed >> 8)  & 1; C[i * K + base_j + 8]  += a_val * bits8;
                uint32_t bits9  = (packed >> 9)  & 1; C[i * K + base_j + 9]  += a_val * bits9;
                uint32_t bits10 = (packed >> 10) & 1; C[i * K + base_j + 10] += a_val * bits10;
                uint32_t bits11 = (packed >> 11) & 1; C[i * K + base_j + 11] += a_val * bits11;
                uint32_t bits12 = (packed >> 12) & 1; C[i * K + base_j + 12] += a_val * bits12;
                uint32_t bits13 = (packed >> 13) & 1; C[i * K + base_j + 13] += a_val * bits13;
                uint32_t bits14 = (packed >> 14) & 1; C[i * K + base_j + 14] += a_val * bits14;
                uint32_t bits15 = (packed >> 15) & 1; C[i * K + base_j + 15] += a_val * bits15;
                uint32_t bits16 = (packed >> 16) & 1; C[i * K + base_j + 16] += a_val * bits16;
                uint32_t bits17 = (packed >> 17) & 1; C[i * K + base_j + 17] += a_val * bits17;
                uint32_t bits18 = (packed >> 18) & 1; C[i * K + base_j + 18] += a_val * bits18;
                uint32_t bits19 = (packed >> 19) & 1; C[i * K + base_j + 19] += a_val * bits19;
                uint32_t bits20 = (packed >> 20) & 1; C[i * K + base_j + 20] += a_val * bits20;
                uint32_t bits21 = (packed >> 21) & 1; C[i * K + base_j + 21] += a_val * bits21;
                uint32_t bits22 = (packed >> 22) & 1; C[i * K + base_j + 22] += a_val * bits22;
                uint32_t bits23 = (packed >> 23) & 1; C[i * K + base_j + 23] += a_val * bits23;
                uint32_t bits24 = (packed >> 24) & 1; C[i * K + base_j + 24] += a_val * bits24;
                uint32_t bits25 = (packed >> 25) & 1; C[i * K + base_j + 25] += a_val * bits25;
                uint32_t bits26 = (packed >> 26) & 1; C[i * K + base_j + 26] += a_val * bits26;
                uint32_t bits27 = (packed >> 27) & 1; C[i * K + base_j + 27] += a_val * bits27;
                uint32_t bits28 = (packed >> 28) & 1; C[i * K + base_j + 28] += a_val * bits28;
                uint32_t bits29 = (packed >> 29) & 1; C[i * K + base_j + 29] += a_val * bits29;
                uint32_t bits30 = (packed >> 30) & 1; C[i * K + base_j + 30] += a_val * bits30;
                uint32_t bits31 = (packed >> 31) & 1; C[i * K + base_j + 31] += a_val * bits31;
            }
        }
        // Final conversion: C[i,j] = 2.0f * C2[i,j] - row_sum[i]
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 2.0f * C[i * K + j] - rsum;
        }
    }
}