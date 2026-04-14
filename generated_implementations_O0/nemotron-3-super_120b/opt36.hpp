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

            // Store a_vals in an array for easier indexing in the inner loop
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
                // Process each bit in the chunk (0..31) unrolled by 8
                for (size_t b = 0; b < 32; b += 8) {
                    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
                    float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
                    // ii = 0
                    uint32_t bit0  = (p0  >> (b+0)) & 1; sum0  += a_vals[0] * bit0;
                    uint32_t bit1  = (p1  >> (b+0)) & 1; sum1  += a_vals[1] * bit1;
                    uint32_t bit2  = (p2  >> (b+0)) & 1; sum2  += a_vals[2] * bit2;
                    uint32_t bit3  = (p3  >> (b+0)) & 1; sum3  += a_vals[3] * bit3;
                    uint32_t bit4  = (p4  >> (b+0)) & 1; sum4  += a_vals[4] * bit4;
                    uint32_t bit5  = (p5  >> (b+0)) & 1; sum5  += a_vals[5] * bit5;
                    uint32_t bit6  = (p6  >> (b+0)) & 1; sum6  += a_vals[6] * bit6;
                    uint32_t bit7  = (p7  >> (b+0)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 1
                    bit0  = (p0  >> (b+1)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+1)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+1)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+1)) & 1; sum3  += a_vals[3] * bit3;
                    bit4  = (p4  >> (b+1)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+1)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+1)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+1)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 2
                    bit0  = (p0  >> (b+2)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+2)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+2)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+2)) & 1; sum3  += a_vals[3] * bit3;
                    bit4  = (p4  >> (b+2)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+2)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+2)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+2)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 3
                    bit0  = (p0  >> (b+3)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+3)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+3)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+3)) & 1; sum3  += a_vals[3] * bit3;
                    bit4  = (p4  >> (b+3)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+3)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+3)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+3)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 4
                    bit0  = (p0  >> (b+4)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+4)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+4)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+4)) & 1; sum3  += a_vals[3] * bit3;
                    bit4  = (p4  >> (b+4)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+4)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+4)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+4)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 5
                    bit0  = (p0  >> (b+5)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+5)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+5)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+5)) & 1; sum3  += a_vals[3] * bit3;
                    bit4  = (p4  >> (b+5)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+5)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+5)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+5)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 6
                    bit0  = (p0  >> (b+6)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+6)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+6)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+6)) & 1; sum3  += a_vals[3] * bit3;
                    bit4  = (p4  >> (b+6)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+6)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+6)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+6)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 7
                    bit0  = (p0  >> (b+7)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+7)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+7)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+7)) & 1; sum3  += a_vals[3] * bit3;
                    bit4  = (p4  >> (b+7)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+7)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+7)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+7)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 8
                    bit0  = (p0  >> (b+8)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+8)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+8)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+8)) & 1; sum3  += a_vals[3] * bit3;
                    bit4  = (p4  >> (b+8)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+8)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+8)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+8)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 9
                    bit0  = (p0  >> (b+9)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+9)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+9)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+9)) & 1; sum3  += a_vals[3] * bit3;
                    bit4  = (p4  >> (b+9)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+9)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+9)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+9)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 10
                    bit0  = (p0  >> (b+10)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+10)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+10)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+10)) & 1; sum3  += a_vals[3] * bit3;
                    bit4  = (p4  >> (b+10)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+10)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+10)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+10)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 11
                    bit0  = (p0  >> (b+11)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+11)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+11)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+11)) & 1; sum3  += a_vals[3] * bit3;
                    bit4  = (p4  >> (b+11)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+11)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+11)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+11)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 12
                    bit0  = (p0  >> (b+12)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+12)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+12)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+12)) & 1; sum3  += a_vals[3] * bit3;
                    bit4  = (p4  >> (b+12)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+12)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+12)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+12)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 13
                    bit0  = (p0  >> (b+13)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+13)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+13)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+13)) & 1; sum3  += a_vals[3] * bit3;
                    bit4  = (p4  >> (b+13)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+13)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+13)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+13)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 14
                    bit0  = (p0  >> (b+14)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+14)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+14)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+14)) & 1; sum3  += a_vals[3] * bit3;
                    bit4  = (p4  >> (b+14)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+14)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+14)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+14)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 15
                    bit0  = (p0  >> (b+15)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+15)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+15)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+15)) & 1; sum3  += a_vals[3] * bit3;
                    bit4  = (p4  >> (b+15)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+15)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+15)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+15)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 16
                    bit0  = (p0  >> (b+16)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+16)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+16)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+16)) & 1; sum3  += a_vals[3] * bit3;
                    bit4  = (p4  >> (b+16)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+16)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+16)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+16)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 17
                    bit0  = (p0  >> (b+17)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+17)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+17)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+17)) & 1; sum3  += a_vals[3] * bit3;
                    bit4  = (p4  >> (b+17)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+17)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+17)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+17)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 18
                    bit0  = (p0  >> (b+18)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+18)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+18)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+18)) & 1; sum3  += a_vals[3] * bit3;
                    bit4  = (p4  >> (b+18)) & 1; sum4  = a_vals[4] * bit4;
                    bit5  = (p5  >> (b+18)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+18)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+18)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 19
                    bit0  = (p0  >> (b+19)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+19)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+19)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+19)) & 1; sum3  += a_vals[3] * bit3;
                    bit4  = (p4  >> (b+19)) & 1; sum4  = a_vals[4] * bit4;
                    bit5  = (p5  >> (b+19)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+19)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+19)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 20
                    bit0  = (p0  >> (b+20)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+20)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+20)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+20)) & 1; sum3  = a_vals[3] * bit3;
                    bit4  = (p4  >> (b+20)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+20)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+20)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+20)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 21
                    bit0  = (p0  >> (b+21)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+21)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+21)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+21)) & 1; sum3  = a_vals[3] * bit3;
                    bit4  = (p4  >> (b+21)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+21)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+21)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+21)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 22
                    bit0  = (p0  >> (b+22)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+22)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+22)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+22)) & 1; sum3  = a_vals[3] * bit3;
                    bit4  = (p4  >> (b+22)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+22)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+22)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+22)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 23
                    bit0  = (p0  >> (b+23)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+23)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+23)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+23)) & 1; sum3  = a_vals[3] * bit3;
                    bit4  = (p4  >> (b+23)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+23)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+23)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+23)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 24
                    bit0  = (p0  >> (b+24)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+24)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+24)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+24)) & 1; sum3  = a_vals[3] * bit3;
                    bit4  = (p4  >> (b+24)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+24)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+24)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+24)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 25
                    bit0  = (p0  >> (b+25)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+25)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+25)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+25)) & 1; sum3  = a_vals[3] * bit3;
                    bit4  = (p4  >> (b+25)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+25)) & 1; sum5  += a_vals[5] * bit5;
                    bit6  = (p6  >> (b+25)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+25)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 26
                    bit0  = (p0  >> (b+26)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+26)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+26)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+26)) & 1; sum3  = a_vals[3] * bit3;
                    bit4  = (p4  >> (b+26)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+26)) & 1; sum5  = a_vals[5] * bit5;
                    bit6  = (p6  >> (b+26)) & 1; sum6  += a_vals[6] * bit6;
                    bit7  = (p7  >> (b+26)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 27
                    bit0  = (p0  >> (b+27)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+27)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+27)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+27)) & 1; sum3  = a_vals[3] * bit3;
                    bit4  = (p4  >> (b+27)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+27)) & 1; sum5  = a_vals[5] * bit5;
                    bit6  = (p6  >> (b+27)) & 1; sum6  = a_vals[6] * bit6;
                    bit7  = (p7  >> (b+27)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 28
                    bit0  = (p0  >> (b+28)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+28)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+28)) & 1; sum2  += a_vals[2] * bit2;
                    bit3  = (p3  >> (b+28)) & 1; sum3  = a_vals[3] * bit3;
                    bit4  = (p4  >> (b+28)) & 1; sum4  += a_vals[4] * bit4;
                    bit5  = (p5  >> (b+28)) & 1; sum5  = a_vals[5] * bit5;
                    bit6  = (p6  >> (b+28)) & 1; sum6  = a_vals[6] * bit6;
                    bit7  = (p7  >> (b+28)) & 1; sum7  += a_vals[7] * bit7;
                    // ii = 29
                    bit0  = (p0  >> (b+29)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+29)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+29)) & 1; sum2  = a_vals[2] * bit2;
                    bit3  = (p3  >> (b+29)) & 1; sum3  = a_vals[3] * bit3;
                    bit4  = (p4  >> (b+29)) & 1; sum4  = a_vals[4] * bit4;
                    bit5  = (p5  >> (b+29)) & 1; sum5  = a_vals[5] * bit5;
                    bit6  = (p6  >> (b+29)) & 1; sum6  = a_vals[6] * bit6;
                    bit7  = (p7  >> (b+29)) & 1; sum7  = a_vals[7] * bit7;
                    // ii = 30
                    bit0  = (p0  >> (b+30)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+30)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+30)) & 1; sum2  = a_vals[2] * bit2;
                    bit3  = (p3  >> (b+30)) & 1; sum3  = a_vals[3] * bit3;
                    bit4  = (p4  >> (b+30)) & 1; sum4  = a_vals[4] * bit4;
                    bit5  = (p5  >> (b+30)) & 1; sum5  = a_vals[5] * bit5;
                    bit6  = (p6  >> (b+30)) & 1; sum6  = a_vals[6] * bit6;
                    bit7  = (p7  >> (b+30)) & 1; sum7  = a_vals[7] * bit7;
                    // ii = 31
                    bit0  = (p0  >> (b+31)) & 1; sum0  += a_vals[0] * bit0;
                    bit1  = (p1  >> (b+31)) & 1; sum1  += a_vals[1] * bit1;
                    bit2  = (p2  >> (b+31)) & 1; sum2  = a_vals[2] * bit2;
                    bit3  = (p3  >> (b+31)) & 1; sum3  = a_vals[3] * bit3;
                    bit4  = (p4  >> (b+31)) & 1; sum4  = a_vals[4] * bit4;
                    bit5  = (p5  >> (b+31)) & 1; sum5  = a_vals[5] * bit5;
                    bit6  = (p6  >> (b+31)) & 1; sum6  = a_vals[6] * bit6;
                    bit7  = (p7  >> (b+31)) & 1; sum7  = a_vals[7] * bit7;
                    // Store the 8 sums
                    C[i * K + base_j + b + 0] += sum0;
                    C[i * K + base_j + b + 1] += sum1;
                    C[i * K + base_j + b + 2] += sum2;
                    C[i * K + base_j + b + 3] += sum3;
                    C[i * K + base_j + b + 4] += sum4;
                    C[i * K + base_j + b + 5] += sum5;
                    C[i * K + base_j + b + 6] += sum6;
                    C[i * K + base_j + b + 7] += sum7;
                }
            }
        }
        // Final conversion: C[i,j] = 2.0f * C2[i,j] - row_sum[i]
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 2.0f * C[i * K + j] - rsum;
        }
    }
}