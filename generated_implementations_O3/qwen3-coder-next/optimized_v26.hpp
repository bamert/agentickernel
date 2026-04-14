#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>

// Optimized matmul with explicit register optimization
// C = A * B where B is a K×K binary matrix packed as 1 bit = ±1.0f

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Initialize C to zero
    memset(C, 0, M * K * sizeof(float));
    
    // Block size for A rows
    const size_t BLOCK_SIZE = 8;
    
    // Process A in blocks of rows
    for (size_t i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
        size_t i1 = (i0 + BLOCK_SIZE <= M) ? i0 + BLOCK_SIZE : M;
        
        // For each row p in B
        for (size_t p = 0; p < K; ++p) {
            // Get the contribution from row p of A for all rows in the block
            float a_vals[8];
            float* a_val_ptr = a_vals;
            for (size_t i = i0; i < i1; ++i) {
                *a_val_ptr++ = A[i * K + p];
            }
            
            // Process row p of B
            const uint32_t* B_row = B + p * K_ints;
            
            // Process each 32-bit block of B
            for (size_t block = 0; block < K_ints; ++block) {
                uint32_t packed = B_row[block];
                size_t base_j = block * 32;
                
                // Precompute signs for this block
                float signs[32];
                for (int k = 0; k < 32; ++k) {
                    float bit_val = (packed >> k) & 1u;
                    signs[k] = bit_val * 2.0f - 1.0f;
                }
                
                // Accumulate contributions to all rows in the block
                float* C_row_i0 = C + i0 * K + base_j;
                float* C_row_i1 = C + (i0+1) * K + base_j;
                float* C_row_i2 = C + (i0+2) * K + base_j;
                float* C_row_i3 = C + (i0+3) * K + base_j;
                float* C_row_i4 = C + (i0+4) * K + base_j;
                float* C_row_i5 = C + (i0+5) * K + base_j;
                float* C_row_i6 = C + (i0+6) * K + base_j;
                float* C_row_i7 = C + (i0+7) * K + base_j;
                
                float a_val0 = a_vals[0];
                float a_val1 = a_vals[1];
                float a_val2 = a_vals[2];
                float a_val3 = a_vals[3];
                float a_val4 = a_vals[4];
                float a_val5 = a_vals[5];
                float a_val6 = a_vals[6];
                float a_val7 = a_vals[7];
                
                // Manual unroll for all 8 rows
                C_row_i0[0] += a_val0 * signs[0];
                C_row_i1[0] += a_val1 * signs[0];
                C_row_i2[0] += a_val2 * signs[0];
                C_row_i3[0] += a_val3 * signs[0];
                C_row_i4[0] += a_val4 * signs[0];
                C_row_i5[0] += a_val5 * signs[0];
                C_row_i6[0] += a_val6 * signs[0];
                C_row_i7[0] += a_val7 * signs[0];
                
                C_row_i0[1] += a_val0 * signs[1];
                C_row_i1[1] += a_val1 * signs[1];
                C_row_i2[1] += a_val2 * signs[1];
                C_row_i3[1] += a_val3 * signs[1];
                C_row_i4[1] += a_val4 * signs[1];
                C_row_i5[1] += a_val5 * signs[1];
                C_row_i6[1] += a_val6 * signs[1];
                C_row_i7[1] += a_val7 * signs[1];
                
                C_row_i0[2] += a_val0 * signs[2];
                C_row_i1[2] += a_val1 * signs[2];
                C_row_i2[2] += a_val2 * signs[2];
                C_row_i3[2] += a_val3 * signs[2];
                C_row_i4[2] += a_val4 * signs[2];
                C_row_i5[2] += a_val5 * signs[2];
                C_row_i6[2] += a_val6 * signs[2];
                C_row_i7[2] += a_val7 * signs[2];
                
                C_row_i0[3] += a_val0 * signs[3];
                C_row_i1[3] += a_val1 * signs[3];
                C_row_i2[3] += a_val2 * signs[3];
                C_row_i3[3] += a_val3 * signs[3];
                C_row_i4[3] += a_val4 * signs[3];
                C_row_i5[3] += a_val5 * signs[3];
                C_row_i6[3] += a_val6 * signs[3];
                C_row_i7[3] += a_val7 * signs[3];
                
                C_row_i0[4] += a_val0 * signs[4];
                C_row_i1[4] += a_val1 * signs[4];
                C_row_i2[4] += a_val2 * signs[4];
                C_row_i3[4] += a_val3 * signs[4];
                C_row_i4[4] += a_val4 * signs[4];
                C_row_i5[4] += a_val5 * signs[4];
                C_row_i6[4] += a_val6 * signs[4];
                C_row_i7[4] += a_val7 * signs[4];
                
                C_row_i0[5] += a_val0 * signs[5];
                C_row_i1[5] += a_val1 * signs[5];
                C_row_i2[5] += a_val2 * signs[5];
                C_row_i3[5] += a_val3 * signs[5];
                C_row_i4[5] += a_val4 * signs[5];
                C_row_i5[5] += a_val5 * signs[5];
                C_row_i6[5] += a_val6 * signs[5];
                C_row_i7[5] += a_val7 * signs[5];
                
                C_row_i0[6] += a_val0 * signs[6];
                C_row_i1[6] += a_val1 * signs[6];
                C_row_i2[6] += a_val2 * signs[6];
                C_row_i3[6] += a_val3 * signs[6];
                C_row_i4[6] += a_val4 * signs[6];
                C_row_i5[6] += a_val5 * signs[6];
                C_row_i6[6] += a_val6 * signs[6];
                C_row_i7[6] += a_val7 * signs[6];
                
                C_row_i0[7] += a_val0 * signs[7];
                C_row_i1[7] += a_val1 * signs[7];
                C_row_i2[7] += a_val2 * signs[7];
                C_row_i3[7] += a_val3 * signs[7];
                C_row_i4[7] += a_val4 * signs[7];
                C_row_i5[7] += a_val5 * signs[7];
                C_row_i6[7] += a_val6 * signs[7];
                C_row_i7[7] += a_val7 * signs[7];
                
                C_row_i0[8] += a_val0 * signs[8];
                C_row_i1[8] += a_val1 * signs[8];
                C_row_i2[8] += a_val2 * signs[8];
                C_row_i3[8] += a_val3 * signs[8];
                C_row_i4[8] += a_val4 * signs[8];
                C_row_i5[8] += a_val5 * signs[8];
                C_row_i6[8] += a_val6 * signs[8];
                C_row_i7[8] += a_val7 * signs[8];
                
                C_row_i0[9] += a_val0 * signs[9];
                C_row_i1[9] += a_val1 * signs[9];
                C_row_i2[9] += a_val2 * signs[9];
                C_row_i3[9] += a_val3 * signs[9];
                C_row_i4[9] += a_val4 * signs[9];
                C_row_i5[9] += a_val5 * signs[9];
                C_row_i6[9] += a_val6 * signs[9];
                C_row_i7[9] += a_val7 * signs[9];
                
                C_row_i0[10] += a_val0 * signs[10];
                C_row_i1[10] += a_val1 * signs[10];
                C_row_i2[10] += a_val2 * signs[10];
                C_row_i3[10] += a_val3 * signs[10];
                C_row_i4[10] += a_val4 * signs[10];
                C_row_i5[10] += a_val5 * signs[10];
                C_row_i6[10] += a_val6 * signs[10];
                C_row_i7[10] += a_val7 * signs[10];
                
                C_row_i0[11] += a_val0 * signs[11];
                C_row_i1[11] += a_val1 * signs[11];
                C_row_i2[11] += a_val2 * signs[11];
                C_row_i3[11] += a_val3 * signs[11];
                C_row_i4[11] += a_val4 * signs[11];
                C_row_i5[11] += a_val5 * signs[11];
                C_row_i6[11] += a_val6 * signs[11];
                C_row_i7[11] += a_val7 * signs[11];
                
                C_row_i0[12] += a_val0 * signs[12];
                C_row_i1[12] += a_val1 * signs[12];
                C_row_i2[12] += a_val2 * signs[12];
                C_row_i3[12] += a_val3 * signs[12];
                C_row_i4[12] += a_val4 * signs[12];
                C_row_i5[12] += a_val5 * signs[12];
                C_row_i6[12] += a_val6 * signs[12];
                C_row_i7[12] += a_val7 * signs[12];
                
                C_row_i0[13] += a_val0 * signs[13];
                C_row_i1[13] += a_val1 * signs[13];
                C_row_i2[13] += a_val2 * signs[13];
                C_row_i3[13] += a_val3 * signs[13];
                C_row_i4[13] += a_val4 * signs[13];
                C_row_i5[13] += a_val5 * signs[13];
                C_row_i6[13] += a_val6 * signs[13];
                C_row_i7[13] += a_val7 * signs[13];
                
                C_row_i0[14] += a_val0 * signs[14];
                C_row_i1[14] += a_val1 * signs[14];
                C_row_i2[14] += a_val2 * signs[14];
                C_row_i3[14] += a_val3 * signs[14];
                C_row_i4[14] += a_val4 * signs[14];
                C_row_i5[14] += a_val5 * signs[14];
                C_row_i6[14] += a_val6 * signs[14];
                C_row_i7[14] += a_val7 * signs[14];
                
                C_row_i0[15] += a_val0 * signs[15];
                C_row_i1[15] += a_val1 * signs[15];
                C_row_i2[15] += a_val2 * signs[15];
                C_row_i3[15] += a_val3 * signs[15];
                C_row_i4[15] += a_val4 * signs[15];
                C_row_i5[15] += a_val5 * signs[15];
                C_row_i6[15] += a_val6 * signs[15];
                C_row_i7[15] += a_val7 * signs[15];
                
                C_row_i0[16] += a_val0 * signs[16];
                C_row_i1[16] += a_val1 * signs[16];
                C_row_i2[16] += a_val2 * signs[16];
                C_row_i3[16] += a_val3 * signs[16];
                C_row_i4[16] += a_val4 * signs[16];
                C_row_i5[16] += a_val5 * signs[16];
                C_row_i6[16] += a_val6 * signs[16];
                C_row_i7[16] += a_val7 * signs[16];
                
                C_row_i0[17] += a_val0 * signs[17];
                C_row_i1[17] += a_val1 * signs[17];
                C_row_i2[17] += a_val2 * signs[17];
                C_row_i3[17] += a_val3 * signs[17];
                C_row_i4[17] += a_val4 * signs[17];
                C_row_i5[17] += a_val5 * signs[17];
                C_row_i6[17] += a_val6 * signs[17];
                C_row_i7[17] += a_val7 * signs[17];
                
                C_row_i0[18] += a_val0 * signs[18];
                C_row_i1[18] += a_val1 * signs[18];
                C_row_i2[18] += a_val2 * signs[18];
                C_row_i3[18] += a_val3 * signs[18];
                C_row_i4[18] += a_val4 * signs[18];
                C_row_i5[18] += a_val5 * signs[18];
                C_row_i6[18] += a_val6 * signs[18];
                C_row_i7[18] += a_val7 * signs[18];
                
                C_row_i0[19] += a_val0 * signs[19];
                C_row_i1[19] += a_val1 * signs[19];
                C_row_i2[19] += a_val2 * signs[19];
                C_row_i3[19] += a_val3 * signs[19];
                C_row_i4[19] += a_val4 * signs[19];
                C_row_i5[19] += a_val5 * signs[19];
                C_row_i6[19] += a_val6 * signs[19];
                C_row_i7[19] += a_val7 * signs[19];
                
                C_row_i0[20] += a_val0 * signs[20];
                C_row_i1[20] += a_val1 * signs[20];
                C_row_i2[20] += a_val2 * signs[20];
                C_row_i3[20] += a_val3 * signs[20];
                C_row_i4[20] += a_val4 * signs[20];
                C_row_i5[20] += a_val5 * signs[20];
                C_row_i6[20] += a_val6 * signs[20];
                C_row_i7[20] += a_val7 * signs[20];
                
                C_row_i0[21] += a_val0 * signs[21];
                C_row_i1[21] += a_val1 * signs[21];
                C_row_i2[21] += a_val2 * signs[21];
                C_row_i3[21] += a_val3 * signs[21];
                C_row_i4[21] += a_val4 * signs[21];
                C_row_i5[21] += a_val5 * signs[21];
                C_row_i6[21] += a_val6 * signs[21];
                C_row_i7[21] += a_val7 * signs[21];
                
                C_row_i0[22] += a_val0 * signs[22];
                C_row_i1[22] += a_val1 * signs[22];
                C_row_i2[22] += a_val2 * signs[22];
                C_row_i3[22] += a_val3 * signs[22];
                C_row_i4[22] += a_val4 * signs[22];
                C_row_i5[22] += a_val5 * signs[22];
                C_row_i6[22] += a_val6 * signs[22];
                C_row_i7[22] += a_val7 * signs[22];
                
                C_row_i0[23] += a_val0 * signs[23];
                C_row_i1[23] += a_val1 * signs[23];
                C_row_i2[23] += a_val2 * signs[23];
                C_row_i3[23] += a_val3 * signs[23];
                C_row_i4[23] += a_val4 * signs[23];
                C_row_i5[23] += a_val5 * signs[23];
                C_row_i6[23] += a_val6 * signs[23];
                C_row_i7[23] += a_val7 * signs[23];
                
                C_row_i0[24] += a_val0 * signs[24];
                C_row_i1[24] += a_val1 * signs[24];
                C_row_i2[24] += a_val2 * signs[24];
                C_row_i3[24] += a_val3 * signs[24];
                C_row_i4[24] += a_val4 * signs[24];
                C_row_i5[24] += a_val5 * signs[24];
                C_row_i6[24] += a_val6 * signs[24];
                C_row_i7[24] += a_val7 * signs[24];
                
                C_row_i0[25] += a_val0 * signs[25];
                C_row_i1[25] += a_val1 * signs[25];
                C_row_i2[25] += a_val2 * signs[25];
                C_row_i3[25] += a_val3 * signs[25];
                C_row_i4[25] += a_val4 * signs[25];
                C_row_i5[25] += a_val5 * signs[25];
                C_row_i6[25] += a_val6 * signs[25];
                C_row_i7[25] += a_val7 * signs[25];
                
                C_row_i0[26] += a_val0 * signs[26];
                C_row_i1[26] += a_val1 * signs[26];
                C_row_i2[26] += a_val2 * signs[26];
                C_row_i3[26] += a_val3 * signs[26];
                C_row_i4[26] += a_val4 * signs[26];
                C_row_i5[26] += a_val5 * signs[26];
                C_row_i6[26] += a_val6 * signs[26];
                C_row_i7[26] += a_val7 * signs[26];
                
                C_row_i0[27] += a_val0 * signs[27];
                C_row_i1[27] += a_val1 * signs[27];
                C_row_i2[27] += a_val2 * signs[27];
                C_row_i3[27] += a_val3 * signs[27];
                C_row_i4[27] += a_val4 * signs[27];
                C_row_i5[27] += a_val5 * signs[27];
                C_row_i6[27] += a_val6 * signs[27];
                C_row_i7[27] += a_val7 * signs[27];
                
                C_row_i0[28] += a_val0 * signs[28];
                C_row_i1[28] += a_val1 * signs[28];
                C_row_i2[28] += a_val2 * signs[28];
                C_row_i3[28] += a_val3 * signs[28];
                C_row_i4[28] += a_val4 * signs[28];
                C_row_i5[28] += a_val5 * signs[28];
                C_row_i6[28] += a_val6 * signs[28];
                C_row_i7[28] += a_val7 * signs[28];
                
                C_row_i0[29] += a_val0 * signs[29];
                C_row_i1[29] += a_val1 * signs[29];
                C_row_i2[29] += a_val2 * signs[29];
                C_row_i3[29] += a_val3 * signs[29];
                C_row_i4[29] += a_val4 * signs[29];
                C_row_i5[29] += a_val5 * signs[29];
                C_row_i6[29] += a_val6 * signs[29];
                C_row_i7[29] += a_val7 * signs[29];
                
                C_row_i0[30] += a_val0 * signs[30];
                C_row_i1[30] += a_val1 * signs[30];
                C_row_i2[30] += a_val2 * signs[30];
                C_row_i3[30] += a_val3 * signs[30];
                C_row_i4[30] += a_val4 * signs[30];
                C_row_i5[30] += a_val5 * signs[30];
                C_row_i6[30] += a_val6 * signs[30];
                C_row_i7[30] += a_val7 * signs[30];
                
                C_row_i0[31] += a_val0 * signs[31];
                C_row_i1[31] += a_val1 * signs[31];
                C_row_i2[31] += a_val2 * signs[31];
                C_row_i3[31] += a_val3 * signs[31];
                C_row_i4[31] += a_val4 * signs[31];
                C_row_i5[31] += a_val5 * signs[31];
                C_row_i6[31] += a_val6 * signs[31];
                C_row_i7[31] += a_val7 * signs[31];
            }
        }
    }
}
