#pragma once
#include <cstdint>
#include <cstddev>

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
            float a_vals[16];
            for (size_t ii = 0; ii < 16; ++ii) {
                a_vals[ii] = A[i * K + p_start + ii];
                rsum += a_vals[ii];
            }

            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                size_t base_j = chunk_idx * 32;
                // Load 16 packed values (one for each of the 16 rows, for this chunk)
                uint32_t packed_vals[16];
                for (size_t ii = 0; ii < 16; ++ii) {
                    packed_vals[ii] = B[(p_start + ii) * K_ints + chunk_idx];
                }
                // Process each bit in the chunk (0..31)
                for (size_t b = 0; b < 32; ++b) {
                    float sum = 0.0f;
                    // Unroll the ii loop
                    sum += a_vals[0] * ((packed_vals[0] >> b) & 1u);
                    sum += a_vals[1] * ((packed_vals[1] >> b) & 1u);
                    sum += a_vals[2] * ((packed_vals[2] >> b) & 1u);
                    sum += a_vals[3] * ((packed_vals[3] >> b) & 1u);
                    sum += a_vals[4] * ((packed_vals[4] >> b) & 1u);
                    sum += a_vals[5] * ((packed_vals[5] >> b) & 1u);
                    sum += a_vals[6] * ((packed_vals[6] >> b) & 1u);
                    sum += a_vals[7] * ((packed_vals[7] >> b) & 1u);
                    sum += a_vals[8] * ((packed_vals[8] >> b) & 1u);
                    sum += a_vals[9] * ((packed_vals[9] >> b) & 1u);
                    sum += a_vals[10] * ((packed_vals[10] >> b) & 1u);
                    sum += a_vals[11] * ((packed_vals[11] >> b) & 1u);
                    sum += a_vals[12] * ((packed_vals[12] >> b) & 1u);
                    sum += a_vals[13] * ((packed_vals[13] >> b) & 1u);
                    sum += a_vals[14] * ((packed_vals[14] >> b) & 1u);
                    sum += a_vals[15] * ((packed_vals[15] >> b) & 1u);
                    C[i * K + base_j + b] += sum;
                }
            }
        }
        // Remainder loop for p (should be zero because K is multiple of 32 and we step by 16)
        for (size_t p = (K / 16) * 16; p < K; ++p) {
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