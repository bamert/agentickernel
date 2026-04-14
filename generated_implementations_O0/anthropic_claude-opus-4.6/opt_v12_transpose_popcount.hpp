
#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// New approach: Transpose B first, then for each (i,j), compute dot product
// of A_row[i] with B_transposed_row[j].
// B_transposed[j][p_word] has bit p%32 set if B[p][j] had that bit set.
// 
// Actually, B is K×K packed. B[p] row has K_ints uint32s.
// B_T[j] row would have K_ints uint32s where bit p%32 of word p/32 = bit j%32 of B[p][j/32].
//
// With B transposed, C[i][j] = sum_p A[i][p] * sign(B_T[j][p])
// This is the same computation but now B_T[j] is a contiguous row.
//
// Better approach: quantize A rows to compute partial sums.
// For 32 consecutive p values, we have one uint32 from B_T[j].
// The positive sum = sum of A[i][p] where bit is 1.
// We need a way to selectively sum floats based on bits.
//
// Actually, let me just try the transpose + column-major approach.
// The key benefit: instead of random column access in B, we get row access in B_T.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Transpose B: B is K rows × K_ints uint32s
    // B_T will be K rows × K_ints uint32s (it's K×K so transpose is same size)
    // B_T[j][p/32] bit (p%32) = B[p][j/32] bit (j%32)
    
    // Allocate B_T on stack or heap
    // K×K_ints uint32s = K*K/32 uint32s. For K=3072: 3072*96 = 294912 uint32s = ~1.1MB
    // Too big for stack, use heap
    uint32_t* B_T = new uint32_t[K * K_ints];
    
    // Initialize to zero
    for (size_t idx = 0; idx < K * K_ints; ++idx) {
        B_T[idx] = 0;
    }
    
    // Transpose: for each row p of B, for each group g, for each bit b
    // B[p][g] bit b -> B_T[g*32+b][p/32] bit (p%32)
    for (size_t p = 0; p < K; ++p) {
        size_t p_word = p / 32;
        uint32_t p_bit = 1u << (p % 32);
        const uint32_t* B_row = B + p * K_ints;
        
        for (size_t g = 0; g < K_ints; ++g) {
            uint32_t packed = B_row[g];
            size_t j_base = g * 32;
            
            while (packed) {
                int b = __builtin_ctz(packed);
                B_T[(j_base + b) * K_ints + p_word] |= p_bit;
                packed &= packed - 1; // clear lowest set bit
            }
        }
    }
    
    // Now compute C[i][j] = sum_p A[i][p] * sign(B_T[j][p])
    // For each row i, for each output j, iterate over p in groups of 32
    // For a group, B_T[j][p_word] gives us 32 signs.
    // C[i][j] = sum over p_word { sum of A[i][p_word*32+b] * sign(bit b of B_T[j][p_word]) }
    //         = sum over p_word { 2 * (sum of A where bit=1) - (sum of all A in group) }
    //
    // Precompute partial sums of A rows in groups of 32:
    // A_group_sum[i][p_word] = sum of A[i][p_word*32 .. p_word*32+31]
    
    // For each row of A, precompute group sums
    // Then for each j, use popcount-like approach... but we need weighted sums not counts.
    
    // Actually the simplest win: just do the multiply with transposed B using the 
    // scatter-accumulate approach (same as v7 but on B_T)
    
    // Better: for each (i, j), process all p. With B_T, B_T[j] is contiguous.
    // Process 4 j values at a time to reuse A reads.
    
    size_t i = 0;
    for (; i + 4 <= M; i += 4) {
        const float* A_row0 = A + (i + 0) * K;
        const float* A_row1 = A + (i + 1) * K;
        const float* A_row2 = A + (i + 2) * K;
        const float* A_row3 = A + (i + 3) * K;
        float* C_row0 = C + (i + 0) * K;
        float* C_row1 = C + (i + 1) * K;
        float* C_row2 = C + (i + 2) * K;
        float* C_row3 = C + (i + 3) * K;
        
        for (size_t j = 0; j < K; ++j) {
            const uint32_t* BT_row = B_T + j * K_ints;
            float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
            
            for (size_t pw = 0; pw < K_ints; ++pw) {
                uint32_t packed = BT_row[pw];
                const float* a0 = A_row0 + pw * 32;
                const float* a1 = A_row1 + pw * 32;
                const float* a2 = A_row2 + pw * 32;
                const float* a3 = A_row3 + pw * 32;
                
                for (int b = 0; b < 32; ++b) {
                    float sign = (packed & (1u << b)) ? 1.0f : -1.0f;
                    sum0 += a0[b] * sign;
                    sum1 += a1[b] * sign;
                    sum2 += a2[b] * sign;
                    sum3 += a3[b] * sign;
                }
            }
            
            C_row0[j] = sum0;
            C_row1[j] = sum1;
            C_row2[j] = sum2;
            C_row3[j] = sum3;
        }
    }
    
    for (; i < M; ++i) {
        const float* A_row = A + i * K;
        float* C_row = C + i * K;
        
        for (size_t j = 0; j < K; ++j) {
            const uint32_t* BT_row = B_T + j * K_ints;
            float sum = 0;
            
            for (size_t pw = 0; pw < K_ints; ++pw) {
                uint32_t packed = BT_row[pw];
                const float* a = A_row + pw * 32;
                
                for (int b = 0; b < 32; ++b) {
                    float sign = (packed & (1u << b)) ? 1.0f : -1.0f;
                    sum += a[b] * sign;
                }
            }
            
            C_row[j] = sum;
        }
    }
    
    delete[] B_T;
}
