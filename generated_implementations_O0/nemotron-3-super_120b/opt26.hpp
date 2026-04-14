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
        size_t p = 0;
        // Process 16 p's at a time
        for (; p <= K - 16; p += 16) {
            // Load 16 a_vals
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
            rsum += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 +
                    a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;

            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                size_t base_j = chunk_idx * 32;
                // Load 16 packed values for this chunk
                uint32_t packed0 = B[(p+0) * K_ints + chunk_idx];
                uint32_t packed1 = B[(p+1) * K_ints + chunk_idx];
                uint32_t packed2 = B[(p+2) * K_ints + chunk_idx];
                uint32_t packed3 = B[(p+3) * K_ints + chunk_idx];
                uint32_t packed4 = B[(p+4) * K_ints + chunk_idx];
                uint32_t packed5 = B[(p+5) * K_ints + chunk_idx];
                uint32_t packed6 = B[(p+6) * K_ints + chunk_idx];
                uint32_t packed7 = B[(p+7) * K_ints + chunk_idx];
                uint32_t packed8 = B[(p+8) * K_ints + chunk_idx];
                uint32_t packed9 = B[(p+9) * K_ints + chunk_idx];
                uint32_t packed10 = B[(p+10) * K_ints + chunk_idx];
                uint32_t packed11 = B[(p+11) * K_ints + chunk_idx];
                uint32_t packed12 = B[(p+12) * K_ints + chunk_idx];
                uint32_t packed13 = B[(p+13) * K_ints + chunk_idx];
                uint32_t packed14 = B[(p+14) * K_ints + chunk_idx];
                uint32_t packed15 = B[(p+15) * K_ints + chunk_idx];
                // Process each bit in the chunk (0..31) unrolled by 8
                for (size_t b = 0; b < 32; b += 8) {
                    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
                    float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
                    // ii = 0
                    uint32_t bits8 = (packed0 >> b) & 0xFF;
                    if (bits8 & 0x01) sum0 += a0;
                    if (bits8 & 0x02) sum1 += a0;
                    if (bits8 & 0x04) sum2 += a0;
                    if (bits8 & 0x08) sum3 += a0;
                    if (bits8 & 0x10) sum4 += a0;
                    if (bits8 & 0x20) sum5 += a0;
                    if (bits8 & 0x40) sum6 += a0;
                    if (bits8 & 0x80) sum7 += a0;
                    // ii = 1
                    bits8 = (packed1 >> b) & 0xFF;
                    if (bits8 & 0x01) sum0 += a1;
                    if (bits8 & 0x02) sum1 += a1;
                    if (bits8 & 0x04) sum2 += a1;
                    if (bits8 & 0x08) sum3 += a1;
                    if (bits8 & 0x10) sum4 += a1;
                    if (bits8 & 0x20) sum5 += a1;
                    if (bits8 & 0x40) sum6 += a1;
                    if (bits8 & 0x80) sum7 += a1;
                    // ii = 2
                    bits8 = (packed2 >> b) & 0xFF;
                    if (bits8 & 0x01) sum0 += a2;
                    if (bits8 & 0x02) sum1 += a2;
                    if (bits8 & 0x04) sum2 += a2;
                    if (bits8 & 0x08) sum3 += a2;
                    if (bits8 & 0x10) sum4 += a2;
                    if (bits8 & 0x20) sum5 += a2;
                    if (bits8 & 0x40) sum6 += a2;
                    if (bits8 & 0x80) sum7 += a2;
                    // ii = 3
                    bits8 = (packed3 >> b) & 0xFF;
                    if (bits8 & 0x01) sum0 += a3;
                    if (bits8 & 0x02) sum1 += a3;
                    if (bits8 & 0x04) sum2 += a3;
                    if (bits8 & 0x08) sum3 += a3;
                    if (bits8 & 0x10) sum4 += a3;
                    if (bits8 & 0x20) sum5 += a3;
                    if (bits8 & 0x40) sum6 += a3;
                    if (bits8 & 0x80) sum7 += a3;
                    // ii = 4
                    bits8 = (packed4 >> b) & 0xFF;
                    if (bits8 & 0x01) sum0 += a4;
                    if (bits8 & 0x02) sum1 += a4;
                    if (bits8 & 0x04) sum2 += a4;
                    if (bits8 & 0x08) sum3 += a4;
                    if (bits8 & 0x10) sum4 += a4;
                    if (bits8 & 0x20) sum5 += a4;
                    if (bits8 & 0x40) sum6 += a4;
                    if (bits8 & 0x80) sum7 += a4;
                    // ii = 5
                    bits8 = (packed5 >> b) & 0xFF;
                    if (bits8 & 0x01) sum0 += a5;
                    if (bits8 & 0x02) sum1 += a5;
                    if (bits8 & 0x04) sum2 += a5;
                    if (bits8 & 0x08) sum3 += a5;
                    if (bits8 & 0x10) sum4 += a5;
                    if (bits8 & 0x20) sum5 += a5;
                    if (bits8 & 0x40) sum6 += a5;
                    if (bits8 & 0x80) sum7 += a5;
                    // ii = 6
                    bits8 = (packed6 >> b) & 0xFF;
                    if (bits8 & 0x01) sum0 += a6;
                    if (bits8 & 0x02) sum1 += a6;
                    if (bits8 & 0x04) sum2 += a6;
                    if (bits8 & 0x08) sum3 += a6;
                    if (bits8 & 0x10) sum4 += a6;
                    if (bits8 & 0x20) sum5 += a6;
                    if (bits8 & 0x40) sum6 += a6;
                    if (bits8 & 0x80) sum7 += a6;
                    // ii = 7
                    bits8 = (packed7 >> b) & 0xFF;
                    if (bits8 & 0x01) sum0 += a7;
                    if (bits8 & 0x02) sum1 += a7;
                    if (bits8 & 0x04) sum2 += a7;
                    if (bits8 & 0x08) sum3 += a7;
                    if (bits8 & 0x10) sum4 += a7;
                    if (bits8 & 0x20) sum5 += a7;
                    if (bits8 & 0x40) sum6 += a7;
                    if (bits8 & 0x80) sum7 += a7;
                    // ii = 8
                    bits8 = (packed8 >> b) & 0xFF;
                    if (bits8 & 0x01) sum0 += a8;
                    if (bits8 & 0x02) sum1 += a8;
                    if (bits8 & 0x04) sum2 += a8;
                    if (bits8 & 0x08) sum3 += a8;
                    if (bits8 & 0x10) sum4 += a8;
                    if (bits8 & 0x20) sum5 += a8;
                    if (bits8 & 0x40) sum6 += a8;
                    if (bits8 & 0x80) sum7 += a8;
                    // ii = 9
                    bits8 = (packed9 >> b) & 0xFF;
                    if (bits8 & 0x01) sum0 += a9;
                    if (bits8 & 0x02) sum1 += a9;
                    if (bits8 & 0x04) sum2 += a9;
                    if (bits8 & 0x08) sum3 += a9;
                    if (bits8 & 0x10) sum4 += a9;
                    if (bits8 & 0x20) sum5 += a9;
                    if (bits8 & 0x40) sum6 += a9;
                    if (bits8 & 0x80) sum7 += a9;
                    // ii = 10
                    bits8 = (packed10 >> b) & 0xFF;
                    if (bits8 & 0x01) sum0 += a10;
                    if (bits8 & 0x02) sum1 += a10;
                    if (bits8 & 0x04) sum2 += a10;
                    if (bits8 & 0x08) sum3 += a10;
                    if (bits8 & 0x10) sum4 += a10;
                    if (bits8 & 0x20) sum5 += a10;
                    if (bits8 & 0x40) sum6 += a10;
                    if (bits8 & 0x80) sum7 += a10;
                    // ii = 11
                    bits8 = (packed11 >> b) & 0xFF;
                    if (bits8 & 0x01) sum0 += a11;
                    if (bits8 & 0x02) sum1 += a11;
                    if (bits8 & 0x04) sum2 += a11;
                    if (bits8 & 0x08) sum3 += a11;
                    if (bits8 & 0x10) sum4 += a11;
                    if (bits8 & 0x20) sum5 += a11;
                    if (bits8 & 0x40) sum6 += a11;
                    if (bits8 & 0x80) sum7 += a11;
                    // ii = 12
                    bits8 = (packed12 >> b) & 0xFF;
                    if (bits8 & 0x01) sum0 += a12;
                    if (bits8 & 0x02) sum1 += a12;
                    if (bits8 & 0x04) sum2 += a12;
                    if (bits8 & 0x08) sum3 += a12;
                    if (bits8 & 0x10) sum4 += a12;
                    if (bits8 & 0x20) sum5 += a12;
                    if (bits8 & 0x40) sum6 += a12;
                    if (bits8 & 0x80) sum7 += a12;
                    // ii = 13
                    bits8 = (packed13 >> b) & 0xFF;
                    if (bits8 & 0x01) sum0 += a13;
                    if (bits8 & 0x02) sum1 += a13;
                    if (bits8 & 0x04) sum2 += a13;
                    if (bits8 & 0x08) sum3 += a13;
                    if (bits8 & 0x10) sum4 += a13;
                    if (bits8 & 0x20) sum5 += a13;
                    if (bits8 & 0x40) sum6 += a13;
                    if (bits8 & 0x80) sum7 += a13;
                    // ii = 14
                    bits8 = (packed14 >> b) & 0xFF;
                    if (bits8 & 0x01) sum0 += a14;
                    if (bits8 & 0x02) sum1 += a14;
                    if (bits8 & 0x04) sum2 += a14;
                    if (bits8 & 0x08) sum3 += a14;
                    if (bits8 & 0x10) sum4 += a14;
                    if (bits8 & 0x20) sum5 += a14;
                    if (bits8 & 0x40) sum6 += a14;
                    if (bits8 & 0x80) sum7 += a14;
                    // ii = 15
                    bits8 = (packed15 >> b) & 0xFF;
                    if (bits8 & 0x01) sum0 += a15;
                    if (bits8 & 0x02) sum1 += a15;
                    if (bits8 & 0x04) sum2 += a15;
                    if (bits8 & 0x08) sum3 += a15;
                    if (bits8 & 0x10) sum4 += a15;
                    if (bits8 & 0x20) sum5 += a15;
                    if (bits8 & 0x40) sum6 += a15;
                    if (bits8 & 0x80) sum7 += a15;
                    // Store the 8 sums
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
        // Remainder loop for p (should not happen because K is multiple of 32 and we step by 16, but keep for safety)
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