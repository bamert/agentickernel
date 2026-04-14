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
        // Process 16 p's at a time
        for (; p <= K - 16; p += 16) {
            // Load 16 a_vals
            float a_val0 = A[i * K + p + 0];
            float a_val1 = A[i * K + p + 1];
            float a_val2 = A[i * K + p + 2];
            float a_val3 = A[i * K + p + 3];
            float a_val4 = A[i * K + p + 4];
            float a_val5 = A[i * K + p + 5];
            float a_val6 = A[i * K + p + 6];
            float a_val7 = A[i * K + p + 7];
            float a_val8 = A[i * K + p + 8];
            float a_val9 = A[i * K + p + 9];
            float a_val10 = A[i * K + p + 10];
            float a_val11 = A[i * K + p + 11];
            float a_val12 = A[i * K + p + 12];
            float a_val13 = A[i * K + p + 13];
            float a_val14 = A[i * K + p + 14];
            float a_val15 = A[i * K + p + 15];
            rsum += a_val0 + a_val1 + a_val2 + a_val3 + a_val4 + a_val5 + a_val6 + a_val7 +
                    a_val8 + a_val9 + a_val10 + a_val11 + a_val12 + a_val13 + a_val14 + a_val15;

            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
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
                size_t base_j = chunk_idx * 32;
                // Process each bit in the chunk (0..31)
                for (size_t b = 0; b < 32; ++b) {
                    uint32_t bit0 = (packed0 >> b) & 1;
                    uint32_t bit1 = (packed1 >> b) & 1;
                    uint32_t bit2 = (packed2 >> b) & 1;
                    uint32_t bit3 = (packed3 >> b) & 1;
                    uint32_t bit4 = (packed4 >> b) & 1;
                    uint32_t bit5 = (packed5 >> b) & 1;
                    uint32_t bit6 = (packed6 >> b) & 1;
                    uint32_t bit7 = (packed7 >> b) & 1;
                    uint32_t bit8 = (packed8 >> b) & 1;
                    uint32_t bit9 = (packed9 >> b) & 1;
                    uint32_t bit10 = (packed10 >> b) & 1;
                    uint32_t bit11 = (packed11 >> b) & 1;
                    uint32_t bit12 = (packed12 >> b) & 1;
                    uint32_t bit13 = (packed13 >> b) & 1;
                    uint32_t bit14 = (packed14 >> b) & 1;
                    uint32_t bit15 = (packed15 >> b) & 1;
                    C[i * K + base_j + b] += a_val0 * bit0 + a_val1 * bit1 + a_val2 * bit2 + a_val3 * bit3 +
                                             a_val4 * bit4 + a_val5 * bit5 + a_val6 * bit6 + a_val7 * bit7 +
                                             a_val8 * bit8 + a_val9 * bit9 + a_val10 * bit10 + a_val11 * bit11 +
                                             a_val12 * bit12 + a_val13 * bit13 + a_val14 * bit14 + a_val15 * bit15;
                }
            }
        }
        // Remainder loop for p
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