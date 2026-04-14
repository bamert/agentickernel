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

            // Process chunk_idx in steps of 4
            size_t chunk_idx = 0;
            for (; chunk_idx <= K_ints - 4; chunk_idx += 4) {
                size_t base_j0 = chunk_idx * 32;
                size_t base_j1 = (chunk_idx+1) * 32;
                size_t base_j2 = (chunk_idx+2) * 32;
                size_t base_j3 = (chunk_idx+3) * 32;
                // Load 16 packed values for each of the 4 chunks
                uint32_t p0_0 = B[(p+0) * K_ints + chunk_idx + 0];
                uint32_t p1_0 = B[(p+1) * K_ints + chunk_idx + 0];
                uint32_t p2_0 = B[(p+2) * K_ints + chunk_idx + 0];
                uint32_t p3_0 = B[(p+3) * K_ints + chunk_idx + 0];
                uint32_t p4_0 = B[(p+4) * K_ints + chunk_idx + 0];
                uint32_t p5_0 = B[(p+5) * K_ints + chunk_idx + 0];
                uint32_t p6_0 = B[(p+6) * K_ints + chunk_idx + 0];
                uint32_t p7_0 = B[(p+7) * K_ints + chunk_idx + 0];
                uint32_t p8_0 = B[(p+8) * K_ints + chunk_idx + 0];
                uint32_t p9_0 = B[(p+9) * K_ints + chunk_idx + 0];
                uint32_t p10_0 = B[(p+10) * K_ints + chunk_idx + 0];
                uint32_t p11_0 = B[(p+11) * K_ints + chunk_idx + 0];
                uint32_t p12_0 = B[(p+12) * K_ints + chunk_idx + 0];
                uint32_t p13_0 = B[(p+13) * K_ints + chunk_idx + 0];
                uint32_t p14_0 = B[(p+14) * K_ints + chunk_idx + 0];
                uint32_t p15_0 = B[(p+15) * K_ints + chunk_idx + 0];

                uint32_t p0_1 = B[(p+0) * K_ints + chunk_idx + 1];
                uint32_t p1_1 = B[(p+1) * K_ints + chunk_idx + 1];
                uint32_t p2_1 = B[(p+2) * K_ints + chunk_idx + 1];
                uint32_t p3_1 = B[(p+3) * K_ints + chunk_idx + 1];
                uint32_t p4_1 = B[(p+4) * K_ints + chunk_idx + 1];
                uint32_t p5_1 = B[(p+5) * K_ints + chunk_idx + 1];
                uint32_t p6_1 = B[(p+6) * K_ints + chunk_idx + 1];
                uint32_t p7_1 = B[(p+7) * K_ints + chunk_idx + 1];
                uint32_t p8_1 = B[(p+8) * K_ints + chunk_idx + 1];
                uint32_t p9_1 = B[(p+9) * K_ints + chunk_idx + 1];
                uint32_t p10_1 = B[(p+10) * K_ints + chunk_idx + 1];
                uint32_t p11_1 = B[(p+11) * K_ints + chunk_idx + 1];
                uint32_t p12_1 = B[(p+12) * K_ints + chunk_idx + 1];
                uint32_t p13_1 = B[(p+13) * K_ints + chunk_idx + 1];
                uint32_t p14_1 = B[(p+14) * K_ints + chunk_idx + 1];
                uint32_t p15_1 = B[(p+15) * K_ints + chunk_idx + 1];

                uint32_t p0_2 = B[(p+0) * K_ints + chunk_idx + 2];
                uint32_t p1_2 = B[(p+1) * K_ints + chunk_idx + 2];
                uint32_t p2_2 = B[(p+2) * K_ints + chunk_idx + 2];
                uint32_t p3_2 = B[(p+3) * K_ints + chunk_idx + 2];
                uint32_t p4_2 = B[(p+4) * K_ints + chunk_idx + 2];
                uint32_t p5_2 = B[(p+5) * K_ints + chunk_idx + 2];
                uint32_t p6_2 = B[(p+6) * K_ints + chunk_idx + 2];
                uint32_t p7_2 = B[(p+7) * K_ints + chunk_idx + 2];
                uint32_t p8_2 = B[(p+8) * K_ints + chunk_idx + 2];
                uint32_t p9_2 = B[(p+9) * K_ints + chunk_idx + 2];
                uint32_t p10_2 = B[(p+10) * K_ints + chunk_idx + 2];
                uint32_t p11_2 = B[(p+11) * K_ints + chunk_idx + 2];
                uint32_t p12_2 = B[(p+12) * K_ints + chunk_idx + 2];
                uint32_t p13_2 = B[(p+13) * K_ints + chunk_idx + 2];
                uint32_t p14_2 = B[(p+14) * K_ints + chunk_idx + 2];
                uint32_t p15_2 = B[(p+15) * K_ints + chunk_idx + 2];

                uint32_t p0_3 = B[(p+0) * K_ints + chunk_idx + 3];
                uint32_t p1_3 = B[(p+1) * K_ints + chunk_idx + 3];
                uint32_t p2_3 = B[(p+2) * K_ints + chunk_idx + 3];
                uint32_t p3_3 = B[(p+3) * K_ints + chunk_idx + 3];
                uint32_t p4_3 = B[(p+4) * K_ints + chunk_idx + 3];
                uint32_t p5_3 = B[(p+5) * K_ints + chunk_idx + 3];
                uint32_t p6_3 = B[(p+6) * K_ints + chunk_idx + 3];
                uint32_t p7_3 = B[(p+7) * K_ints + chunk_idx + 3];
                uint32_t p8_3 = B[(p+8) * K_ints + chunk_idx + 3];
                uint32_t p9_3 = B[(p+9) * K_ints + chunk_idx + 3];
                uint32_t p10_3 = B[(p+10) * K_ints + chunk_idx + 3];
                uint32_t p11_3 = B[(p+11) * K_ints + chunk_idx + 3];
                uint32_t p12_3 = B[(p+12) * K_ints + chunk_idx + 3];
                uint32_t p13_3 = B[(p+13) * K_ints + chunk_idx + 3];
                uint32_t p14_3 = B[(p+14) * K_ints + chunk_idx + 3];
                uint32_t p15_3 = B[(p+15) * K_ints + chunk_idx + 3];

                // Process each bit in the chunk (0..31) for the 4 chunks
                for (size_t b = 0; b < 32; ++b) {
                    // Chunk 0
                    uint32_t bit0_0 = (p0_0 >> b) & 1; float sum0_0 = a0  * bit0_0;
                    uint32_t bit1_0 = (p1_0 >> b) & 1; float sum1_0 = a1  * bit1_0;
                    uint32_t bit2_0 = (p2_0 >> b) & 1; float sum2_0 = a2  * bit2_0;
                    uint32_t bit3_0 = (p3_0 >> b) & 1; float sum3_0 = a3  * bit3_0;
                    uint32_t bit4_0 = (p4_0 >> b) & 1; float sum4_0 = a4  * bit4_0;
                    uint32_t bit5_0 = (p5_0 >> b) & 1; float sum5_0 = a5  * bit5_0;
                    uint32_t bit6_0 = (p6_0 >> b) & 1; float sum6_0 = a6  * bit6_0;
                    uint32_t bit7_0 = (p7_0 >> b) & 1; float sum7_0 = a7  * bit7_0;
                    uint32_t bit8_0 = (p8_0 >> b) & 1; float sum8_0 = a8  * bit8_0;
                    uint32_t bit9_0 = (p9_0 >> b) & 1; float sum9_0 = a9  * bit9_0;
                    uint32_t bit10_0 = (p10_0 >> b) & 1; float sum10_0 = a10 * bit10_0;
                    uint32_t bit11_0 = (p11_0 >> b) & 1; float sum11_0 = a11 * bit11_0;
                    uint32_t bit12_0 = (p12_0 >> b) & 1; float sum12_0 = a12 * bit12_0;
                    uint32_t bit13_0 = (p13_0 >> b) & 1; float sum13_0 = a13 * bit13_0;
                    uint32_t bit14_0 = (p14_0 >> b) & 1; float sum14_0 = a14 * bit14_0;
                    uint32_t bit15_0 = (p15_0 >> b) & 1; float sum15_0 = a15 * bit15_0;
                    C[i * K + base_j0 + b] += sum0_0 + sum1_0 + sum2_0 + sum3_0 + sum4_0 + sum5_0 + sum6_0 + sum7_0 +
                                              sum8_0 + sum9_0 + sum10_0 + sum11_0 + sum12_0 + sum13_0 + sum14_0 + sum15_0;
                    // Chunk 1
                    uint32_t bit0_1 = (p0_1 >> b) & 1; float sum0_1 = a0  * bit0_1;
                    uint32_t bit1_1 = (p1_1 >> b) & 1; float sum1_1 = a1  * bit1_1;
                    uint32_t bit2_1 = (p2_1 >> b) & 1; float sum2_1 = a2  * bit2_1;
                    uint32_t bit3_1 = (p3_1 >> b) & 1; float sum3_1 = a3  * bit3_1;
                    uint32_t bit4_1 = (p4_1 >> b) & 1; float sum4_1 = a4  * bit4_1;
                    uint32_t bit5_1 = (p5_1 >> b) & 1; float sum5_1 = a5  * bit5_1;
                    uint32_t bit6_1 = (p6_1 >> b) & 1; float sum6_1 = a6  * bit6_1;
                    uint32_t bit7_1 = (p7_1 >> b) & 1; float sum7_1 = a7  * bit7_1;
                    uint32_t bit8_1 = (p8_1 >> b) & 1; float sum8_1 = a8  * bit8_1;
                    uint32_t bit9_1 = (p9_1 >> b) & 1; float sum9_1 = a9  * bit9_1;
                    uint32_t bit10_1 = (p10_1 >> b) & 1; float sum10_1 = a10 * bit10_1;
                    uint32_t bit11_1 = (p11_1 >> b) & 1; float sum11_1 = a11 * bit11_1;
                    uint32_t bit12_1 = (p12_1 >> b) & 1; float sum12_1 = a12 * bit12_1;
                    uint32_t bit13_1 = (p13_1 >> b) & 1; float sum13_1 = a13 * bit13_1;
                    uint32_t bit14_1 = (p14_1 >> b) & 1; float sum14_1 = a14 * bit14_1;
                    uint32_t bit15_1 = (p15_1 >> b) & 1; float sum15_1 = a15 * bit15_1;
                    C[i * K + base_j1 + b] += sum0_1 + sum1_1 + sum2_1 + sum3_1 + sum4_1 + sum5_1 + sum6_1 + sum7_1 +
                                              sum8_1 + sum9_1 + sum10_1 + sum11_1 + sum12_1 + sum13_1 + sum14_1 + sum15_1;
                    // Chunk 2
                    uint32_t bit0_2 = (p0_2 >> b) & 1; float sum0_2 = a0  * bit0_2;
                    uint32_t bit1_2 = (p1_2 >> b) & 1; float sum1_2 = a1  * bit1_2;
                    uint32_t bit2_2 = (p2_2 >> b) & 1; float sum2_2 = a2  * bit2_2;
                    uint32_t bit3_2 = (p3_2 >> b) & 1; float sum3_2 = a3  * bit3_2;
                    uint32_t bit4_2 = (p4_2 >> b) & 1; float sum4_2 = a4  * bit4_2;
                    uint32_t bit5_2 = (p5_2 >> b) & 1; float sum5_2 = a5  * bit5_2;
                    uint32_t bit6_2 = (p6_2 >> b) & 1; float sum6_2 = a6  * bit6_2;
                    uint32_t bit7_2 = (p7_2 >> b) & 1; float sum7_2 = a7  * bit7_2;
                    uint32_t bit8_2 = (p8_2 >> b) & 1; float sum8_2 = a8  * bit8_2;
                    uint32_t bit9_2 = (p9_2 >> b) & 1; float sum9_2 = a9  * bit9_2;
                    uint32_t bit10_2 = (p10_2 >> b) & 1; float sum10_2 = a10 * bit10_2;
                    uint32_t bit11_2 = (p11_2 >> b) & 1; float sum11_2 = a11 * bit11_2;
                    uint32_t bit12_2 = (p12_2 >> b) & 1; float sum12_2 = a12 * bit12_2;
                    uint32_t bit13_2 = (p13_2 >> b) & 1; float sum13_2 = a13 * bit13_2;
                    uint32_t bit14_2 = (p14_2 >> b) & 1; float sum14_2 = a14 * bit14_2;
                    uint32_t bit15_2 = (p15_2 >> b) & 1; float sum15_2 = a15 * bit15_2;
                    C[i * K + base_j2 + b] += sum0_2 + sum1_2 + sum2_2 + sum3_2 + sum4_2 + sum5_2 + sum6_2 + sum7_2 +
                                              sum8_2 + sum9_2 + sum10_2 + sum11_2 + sum12_2 + sum13_2 + sum14_2 + sum15_2;
                    // Chunk 3
                    uint32_t bit0_3 = (p0_3 >> b) & 1; float sum0_3 = a0  * bit0_3;
                    uint32_t bit1_3 = (p1_3 >> b) & 1; float sum1_3 = a1  * bit1_3;
                    uint32_t bit2_3 = (p2_3 >> b) & 1; float sum2_3 = a2  * bit2_3;
                    uint32_t bit3_3 = (p3_3 >> b) & 1; float sum3_3 = a3  * bit3_3;
                    uint32_t bit4_3 = (p4_3 >> b) & 1; float sum4_3 = a4  * bit4_3;
                    uint32_t bit5_3 = (p5_3 >> b) & 1; float sum5_3 = a5  * bit5_3;
                    uint32_t bit6_3 = (p6_3 >> b) & 1; float sum6_3 = a6  * bit6_3;
                    uint32_t bit7_3 = (p7_3 >> b) & 1; float sum7_3 = a7  * bit7_3;
                    uint32_t bit8_3 = (p8_3 >> b) & 1; float sum8_3 = a8  * bit8_3;
                    uint32_t bit9_3 = (p9_3 >> b) & 1; float sum9_3 = a9  * bit9_3;
                    uint32_t bit10_3 = (p10_3 >> b) & 1; float sum10_3 = a10 * bit10_3;
                    uint32_t bit11_3 = (p11_3 >> b) & 1; float sum11_3 = a11 * bit11_3;
                    uint32_t bit12_3 = (p12_3 >> b) & 1; float sum12_3 = a12 * bit12_3;
                    uint32_t bit13_3 = (p13_3 >> b) & 1; float sum13_3 = a13 * bit13_3;
                    uint32_t bit14_3 = (p14_3 >> b) & 1; float sum14_3 = a14 * bit14_3;
                    uint32_t bit15_3 = (p15_3 >> b) & 1; float sum15_3 = a15 * bit15_3;
                    C[i * K + base_j3 + b] += sum0_3 + sum1_3 + sum2_3 + sum3_3 + sum4_3 + sum5_3 + sum6_3 + sum7_3 +
                                              sum8_3 + sum9_3 + sum10_3 + sum11_3 + sum12_3 + sum13_3 + sum14_3 + sum15_3;
                }
            }
            // Process remaining chunk_idx (less than 4)
            for (; chunk_idx < K_ints; ++chunk_idx) {
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
                // Process each bit in the chunk (0..31)
                for (size_t b = 0; b < 32; ++b) {
                    uint32_t bit0  = (packed0 >> b) & 1; float sum0  = a0  * bit0;
                    uint32_t bit1  = (packed1 >> b) & 1; float sum1  = a1  * bit1;
                    uint32_t bit2  = (packed2 >> b) & 1; float sum2  = a2  * bit2;
                    uint32_t bit3  = (packed3 >> b) & 1; float sum3  = a3  * bit3;
                    uint32_t bit4  = (packed4 >> b) & 1; float sum4  = a4  * bit4;
                    uint32_t bit5  = (packed5 >> b) & 1; float sum5  = a5  * bit5;
                    uint32_t bit6  = (packed6 >> b) & 1; float sum6  = a6  * bit6;
                    uint32_t bit7  = (packed7 >> b) & 1; float sum7  = a7  * bit7;
                    uint32_t bit8  = (packed8 >> b) & 1; float sum8  = a8  * bit8;
                    uint32_t bit9  = (packed9 >> b) & 1; float sum9  = a9  * bit9;
                    uint32_t bit10 = (packed10 >> b) & 1; float sum10 = a10 * bit10;
                    uint32_t bit11 = (packed11 >> b) & 1; float sum11 = a11 * bit11;
                    uint32_t bit12 = (packed12 >> b) & 1; float sum12 = a12 * bit12;
                    uint32_t bit13 = (packed13 >> b) & 1; float sum13 = a13 * bit13;
                    uint32_t bit14 = (packed14 >> b) & 1; float sum14 = a14 * bit14;
                    uint32_t bit15 = (packed15 >> b) & 1; float sum15 = a15 * bit15;
                    C[i * K + base_j + b] += sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 +
                                             sum8 + sum9 + sum10 + sum11 + sum12 + sum13 + sum14 + sum15;
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