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
                    uint32_t bit0  = (p0  >> (b+0)) & 1; float sum0  = a0  * bit0;
                    uint32_t bit1  = (p1  >> (b+0)) & 1; float sum1  = a1  * bit1;
                    uint32_t bit2  = (p2  >> (b+0)) & 1; float sum2  = a2  * bit2;
                    uint32_t bit3  = (p3  >> (b+0)) & 1; float sum3  = a3  * bit3;
                    uint32_t bit4  = (p4  >> (b+0)) & 1; float sum4  = a4  * bit4;
                    uint32_t bit5  = (p5  >> (b+0)) & 1; float sum5  = a5  * bit5;
                    uint32_t bit6  = (p6  >> (b+0)) & 1; float sum6  = a6  * bit6;
                    uint32_t bit7  = (p7  >> (b+0)) & 1; float sum7  = a7  * bit7;
                    uint32_t bit8  = (p8  >> (b+0)) & 1; float sum8  = a8  * bit8;
                    uint32_t bit9  = (p9  >> (b+0)) & 1; float sum9  = a9  * bit9;
                    uint32_t bit10 = (p10 >> (b+0)) & 1; float sum10 = a10 * bit10;
                    uint32_t bit11 = (p11 >> (b+0)) & 1; float sum11 = a11 * bit11;
                    uint32_t bit12 = (p12 >> (b+0)) & 1; float sum12 = a12 * bit12;
                    uint32_t bit13 = (p13 >> (b+0)) & 1; float sum13 = a13 * bit13;
                    uint32_t bit14 = (p14 >> (b+0)) & 1; float sum14 = a14 * bit14;
                    uint32_t bit15 = (p15 >> (b+0)) & 1; float sum15 = a15 * bit15;
                    uint32_t bit16 = (p16 >> (b+0)) & 1; float sum16 = a16 * bit16;
                    uint32_t bit17 = (p17 >> (b+0)) & 1; float sum17 = a17 * bit17;
                    uint32_t bit18 = (p18 >> (b+0)) & 1; float sum18 = a18 * bit18;
                    uint32_t bit19 = (p19 >> (b+0)) & 1; float sum19 = a19 * bit19;
                    uint32_t bit20 = (p20 >> (b+0)) & 1; float sum20 = a20 * bit20;
                    uint32_t bit21 = (p21 >> (b+0)) & 1; float sum21 = a21 * bit21;
                    uint32_t bit22 = (p22 >> (b+0)) & 1; float sum22 = a22 * bit22;
                    uint32_t bit23 = (p23 >> (b+0)) & 1; float sum23 = a23 * bit23;
                    uint32_t bit24 = (p24 >> (b+0)) & 1; float sum24 = a24 * bit24;
                    uint32_t bit25 = (p25 >> (b+0)) & 1; float sum25 = a25 * bit25;
                    uint32_t bit26 = (p26 >> (b+0)) & 1; float sum26 = a26 * bit26;
                    uint32_t bit27 = (p27 >> (b+0)) & 1; float sum27 = a27 * bit27;
                    uint32_t bit28 = (p28 >> (b+0)) & 1; float sum28 = a28 * bit28;
                    uint32_t bit29 = (p29 >> (b+0)) & 1; float sum29 = a29 * bit29;
                    uint32_t bit30 = (p30 >> (b+0)) & 1; float sum30 = a30 * bit30;
                    uint32_t bit31 = (p31 >> (b+0)) & 1; float sum31 = a31 * bit31;
                    // Now do the same for offset b+1 .. b+7? Actually we need to shift for each bit position.
                    // The above only does bit position b for all p's. We need to do for b, b+1, ... b+7.
                    // Let's restructure: we need to compute for each bit position in the chunk.
                    // Instead, we can do inside the b loop: for each offset k=0..7, compute bit at position b+k.
                    // But that would be inner loop. Let's unroll differently: we can compute 8 consecutive bits at once by shifting the packed value by b and masking 0xFF.
                    // Then we can multiply each a_val by the corresponding bit (0 or 1) and add.
                    // We'll do that.
                }
            }
        }
        // Final conversion: C[i,j] = 2.0f * C2[i,j] - row_sum[i]
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 2.0f * C[i * K + j] - rsum;
        }
    }
}