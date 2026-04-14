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
                // Process each bit in the chunk (0..31) unrolled by 8
                for (size_t b = 0; b < 32; b += 8) {
                    // b
                    uint32_t bit0  = (p0  >> b) & 1; float sum0  = a0  * bit0;
                    uint32_t bit1  = (p1  >> b) & 1; float sum1  = a1  * bit1;
                    uint32_t bit2  = (p2  >> b) & 1; float sum2  = a2  * bit2;
                    uint32_t bit3  = (p3  >> b) & 1; float sum3  = a3  * bit3;
                    uint32_t bit4  = (p4  >> b) & 1; float sum4  = a4  * bit4;
                    uint32_t bit5  = (p5  >> b) & 1; float sum5  = a5  * bit5;
                    uint32_t bit6  = (p6  >> b) & 1; float sum6  = a6  * bit6;
                    uint32_t bit7  = (p7  >> b) & 1; float sum7  = a7  * bit7;
                    uint32_t bit8  = (p8  >> b) & 1; float sum8  = a8  * bit8;
                    uint32_t bit9  = (p9  >> b) & 1; float sum9  = a9  * bit9;
                    uint32_t bit10 = (p10 >> b) & 1; float sum10 = a10 * bit10;
                    uint32_t bit11 = (p11 >> b) & 1; float sum11 = a11 * bit11;
                    uint32_t bit12 = (p12 >> b) & 1; float sum12 = a12 * bit12;
                    uint32_t bit13 = (p13 >> b) & 1; float sum13 = a13 * bit13;
                    uint32_t bit14 = (p14 >> b) & 1; float sum14 = a14 * bit14;
                    uint32_t bit15 = (p15 >> b) & 1; float sum15 = a15 * bit15;
                    C[i * K + base_j + b] += sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 +
                                             sum8 + sum9 + sum10 + sum11 + sum12 + sum13 + sum14 + sum15;
                    // b+1
                    bit0  = (p0  >> (b+1)) & 1; sum0  = a0  * bit0;
                    bit1  = (p1  >> (b+1)) & 1; sum1  = a1  * bit1;
                    bit2  = (p2  >> (b+1)) & 1; sum2  = a2  * bit2;
                    bit3  = (p3  >> (b+1)) & 1; sum3  = a3  * bit3;
                    bit4  = (p4  >> (b+1)) & 1; sum4  = a4  * bit4;
                    bit5  = (p5  >> (b+1)) & 1; sum5  = a5  * bit5;
                    bit6  = (p6  >> (b+1)) & 1; sum6  = a6  * bit6;
                    bit7  = (p7  >> (b+1)) & 1; sum7  = a7  * bit7;
                    bit8  = (p8  >> (b+1)) & 1; sum8  = a8  * bit8;
                    bit9  = (p9  >> (b+1)) & 1; sum9  = a9  * bit9;
                    bit10 = (p10 >> (b+1)) & 1; sum10 = a10 * bit10;
                    bit11 = (p11 >> (b+1)) & 1; sum11 = a11 * bit11;
                    bit12 = (p12 >> (b+1)) & 1; sum12 = a12 * bit12;
                    bit13 = (p13 >> (b+1)) & 1; sum13 = a13 * bit13;
                    bit14 = (p14 >> (b+1)) & 1; sum14 = a14 * bit14;
                    bit15 = (p15 >> (b+1)) & 1; sum15 = a15 * bit15;
                    C[i * K + base_j + b+1] += sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 +
                                               sum8 + sum9 + sum10 + sum11 + sum12 + sum13 + sum14 + sum15;
                    // b+2
                    bit0  = (p0  >> (b+2)) & 1; sum0  = a0  * bit0;
                    bit1  = (p1  >> (b+2)) & 1; sum1  = a1  * bit1;
                    bit2  = (p2  >> (b+2)) & 1; sum2  = a2  * bit2;
                    bit3  = (p3  >> (b+2)) & 1; sum3  = a3  * bit3;
                    bit4  = (p4  >> (b+2)) & 1; sum4  = a4  * bit4;
                    bit5  = (p5  >> (b+2)) & 1; sum5  = a5  * bit5;
                    bit6  = (p6  >> (b+2)) & 1; sum6  = a6  * bit6;
                    bit7  = (p7  >> (b+2)) & 1; sum7  = a7  * bit7;
                    bit8  = (p8  >> (b+2)) & 1; sum8  = a8  * bit8;
                    bit9  = (p9  >> (b+2)) & 1; sum9  = a9  * bit9;
                    bit10 = (p10 >> (b+2)) & 1; sum10 = a10 * bit10;
                    bit11 = (p11 >> (b+2)) & 1; sum11 = a11 * bit11;
                    bit12 = (p12 >> (b+2)) & 1; sum12 = a12 * bit12;
                    bit13 = (p13 >> (b+2)) & 1; sum13 = a13 * bit13;
                    bit14 = (p14 >> (b+2)) & 1; sum14 = a14 * bit14;
                    bit15 = (p15 >> (b+2)) & 1; sum15 = a15 * bit15;
                    C[i * K + base_j + b+2] += sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 +
                                               sum8 + sum9 + sum10 + sum11 + sum12 + sum13 + sum14 + sum15;
                    // b+3
                    bit0  = (p0  >> (b+3)) & 1; sum0  = a0  * bit0;
                    bit1  = (p1  >> (b+3)) & 1; sum1  = a1  * bit1;
                    bit2  = (p2  >> (b+3)) & 1; sum2  = a2  * bit2;
                    bit3  = (p3  >> (b+3)) & 1; sum3  = a3  * bit3;
                    bit4  = (p4  >> (b+3)) & 1; sum4  = a4  * bit4;
                    bit5  = (p5  >> (b+3)) & 1; sum5  = a5  * bit5;
                    bit6  = (p6  >> (b+3)) & 1; sum6  = a6  * bit6;
                    bit7  = (p7  >> (b+3)) & 1; sum7  = a7  * bit7;
                    bit8  = (p8  >> (b+3)) & 1; sum8  = a8  * bit8;
                    bit9  = (p9  >> (b+3)) & 1; sum9  = a9  * bit9;
                    bit10 = (p10 >> (b+3)) & 1; sum10 = a10 * bit10;
                    bit11 = (p11 >> (b+3)) & 1; sum11 = a11 * bit11;
                    bit12 = (p12 >> (b+3)) & 1; sum12 = a12 * bit12;
                    bit13 = (p13 >> (b+3)) & 1; sum13 = a13 * bit13;
                    bit14 = (p14 >> (b+3)) & 1; sum14 = a14 * bit14;
                    bit15 = (p15 >> (b+3)) & 1; sum15 = a15 * bit15;
                    C[i * K + base_j + b+3] += sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 +
                                               sum8 + sum9 + sum10 + sum11 + sum12 + sum13 + sum14 + sum15;
                    // b+4
                    bit0  = (p0  >> (b+4)) & 1; sum0  = a0  * bit0;
                    bit1  = (p1  >> (b+4)) & 1; sum1  = a1  * bit1;
                    bit2  = (p2  >> (b+4)) & 1; sum2  = a2  * bit2;
                    bit3  = (p3  >> (b+4)) & 1; sum3  = a3  * bit3;
                    bit4  = (p4  >> (b+4)) & 1; sum4  = a4  * bit4;
                    bit5  = (p5  >> (b+4)) & 1; sum5  = a5  * bit5;
                    bit6  = (p6  >> (b+4)) & 1; sum6  = a6  * bit6;
                    bit7  = (p7  >> (b+4)) & 1; sum7  = a7  * bit7;
                    bit8  = (p8  >> (b+4)) & 1; sum8  = a8  * bit8;
                    bit9  = (p9  >> (b+4)) & 1; sum9  = a9  * bit9;
                    bit10 = (p10 >> (b+4)) & 1; sum10 = a10 * bit10;
                    bit11 = (p11 >> (b+4)) & 1; sum11 = a11 * bit11;
                    bit12 = (p12 >> (b+4)) & 1; sum12 = a12 * bit12;
                    bit13 = (p13 >> (b+4)) & 1; sum13 = a13 * bit13;
                    bit14 = (p14 >> (b+4)) & 1; sum14 = a14 * bit14;
                    bit15 = (p15 >> (b+4)) & 1; sum15 = a15 * bit15;
                    C[i * K + base_j + b+4] += sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 +
                                               sum8 + sum9 + sum10 + sum11 + sum12 + sum13 + sum14 + sum15;
                    // b+5
                    bit0  = (p0  >> (b+5)) & 1; sum0  = a0  * bit0;
                    bit1  = (p1  >> (b+5)) & 1; sum1  = a1  * bit1;
                    bit2  = (p2  >> (b+5)) & 1; sum2  = a2  * bit2;
                    bit3  = (p3  >> (b+5)) & 1; sum3  = a3  * bit3;
                    bit4  = (p4  >> (b+5)) & 1; sum4  = a4  * bit4;
                    bit5  = (p5  >> (b+5)) & 1; sum5  = a5  * bit5;
                    bit6  = (p6  >> (b+5)) & 1; sum6  = a6  * bit6;
                    bit7  = (p7  >> (b+5)) & 1; sum7  = a7  * bit7;
                    bit8  = (p8  >> (b+5)) & 1; sum8  = a8  * bit8;
                    bit9  = (p9  >> (b+5)) & 1; sum9  = a9  * bit9;
                    bit10 = (p10 >> (b+5)) & 1; sum10 = a10 * bit10;
                    bit11 = (p11 >> (b+5)) & 1; sum11 = a11 * bit11;
                    bit12 = (p12 >> (b+5)) & 1; sum12 = a12 * bit12;
                    bit13 = (p13 >> (b+5)) & 1; sum13 = a13 * bit13;
                    bit14 = (p14 >> (b+5)) & 1; sum14 = a14 * bit14;
                    bit15 = (p15 >> (b+5)) & 1; sum15 = a15 * bit15;
                    C[i * K + base_j + b+5] += sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 +
                                               sum8 + sum9 + sum10 + sum11 + sum12 + sum13 + sum14 + sum15;
                    // b+6
                    bit0  = (p0  >> (b+6)) & 1; sum0  = a0  * bit0;
                    bit1  = (p1  >> (b+6)) & 1; sum1  = a1  * bit1;
                    bit2  = (p2  >> (b+6)) & 1; sum2  = a2  * bit2;
                    bit3  = (p3  >> (b+6)) & 1; sum3  = a3  * bit3;
                    bit4  = (p4  >> (b+6)) & 1; sum4  = a4  * bit4;
                    bit5  = (p5  >> (b+6)) & 1; sum5  = a5  * bit5;
                    bit6  = (p6  >> (b+6)) & 1; sum6  = a6  * bit6;
                    bit7  = (p7  >> (b+6)) & 1; sum7  = a7  * bit7;
                    bit8  = (p8  >> (b+6)) & 1; sum8  = a8  * bit8;
                    bit9  = (p9  >> (b+6)) & 1; sum9  = a9  * bit9;
                    bit10 = (p10 >> (b+6)) & 1; sum10 = a10 * bit10;
                    bit11 = (p11 >> (b+6)) & 1; sum11 = a11 * bit11;
                    bit12 = (p12 >> (b+6)) & 1; sum12 = a12 * bit12;
                    bit13 = (p13 >> (b+6)) & 1; sum13 = a13 * bit13;
                    bit14 = (p14 >> (b+6)) & 1; sum14 = a14 * bit14;
                    bit15 = (p15 >> (b+6)) & 1; sum15 = a15 * bit15;
                    C[i * K + base_j + b+6] += sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 +
                                               sum8 + sum9 + sum10 + sum11 + sum12 + sum13 + sum14 + sum15;
                    // b+7
                    bit0  = (p0  >> (b+7)) & 1; sum0  = a0  * bit0;
                    bit1  = (p1  >> (b+7)) & 1; sum1  = a1  * bit1;
                    bit2  = (p2  >> (b+7)) & 1; sum2  = a2  * bit2;
                    bit3  = (p3  >> (b+7)) & 1; sum3  = a3  * bit3;
                    bit4  = (p4  >> (b+7)) & 1; sum4  = a4  * bit4;
                    bit5  = (p5  >> (b+7)) & 1; sum5  = a5  * bit5;
                    bit6  = (p6  >> (b+7)) & 1; sum6  = a6  * bit6;
                    bit7  = (p7  >> (b+7)) & 1; sum7  = a7  * bit7;
                    bit8  = (p8  >> (b+7)) & 1; sum8  = a8  * bit8;
                    bit9  = (p9  >> (b+7)) & 1; sum9  = a9  * bit9;
                    bit10 = (p10 >> (b+7)) & 1; sum10 = a10 * bit10;
                    bit11 = (p11 >> (b+7)) & 1; sum11 = a11 * bit11;
                    bit12 = (p12 >> (b+7)) & 1; sum12 = a12 * bit12;
                    bit13 = (p13 >> (b+7)) & 1; sum13 = a13 * bit13;
                    bit14 = (p14 >> (b+7)) & 1; sum14 = a14 * bit14;
                    bit15 = (p15 >> (b+7)) & 1; sum15 = a15 * bit15;
                    C[i * K + base_j + b+7] += sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 +
                                               sum8 + sum9 + sum10 + sum11 + sum12 + sum13 + sum14 + sum15;
                }
            }
        }
        // Final conversion: C[i,j] = 2.0f * C2[i,j] - row_sum[i]
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 2.0f * C[i * K + j] - rsum;
        }
    }
}