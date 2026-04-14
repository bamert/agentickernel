#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Transpose B: B_T[j][p/32] has bit (p%32) = original B[p][j]
    // B is K rows x K_ints uint32s. B_T is K rows x K_ints uint32s.
    // B_T[j * K_ints + p/32] bit (p%32) = B[p * K_ints + j/32] bit (j%32)
    
    // Allocate B_T on stack or heap
    uint32_t* B_T = (uint32_t*)__builtin_alloca(K * K_ints * sizeof(uint32_t));
    
    // Zero B_T
    for (size_t idx = 0; idx < K * K_ints; ++idx) {
        B_T[idx] = 0;
    }
    
    // Transpose
    for (size_t p = 0; p < K; ++p) {
        size_t p_word = p / 32;
        uint32_t p_bit = 1u << (p % 32);
        for (size_t jb = 0; jb < K_ints; ++jb) {
            uint32_t packed = B[p * K_ints + jb];
            size_t j_base = jb * 32;
            while (packed) {
                int bit = __builtin_ctz(packed);
                size_t j = j_base + bit;
                B_T[j * K_ints + p_word] |= p_bit;
                packed &= packed - 1;
            }
        }
    }
    
    // Now compute C[i][j] = sum over p of A[i][p] * sign(B_T[j][p])
    // sign(bit) = 2*bit - 1, so C[i][j] = 2*popcount(A_pos AND B_T[j]) - based approach
    // 
    // Actually: C[i][j] = sum_p A[i][p] * (2*bit_p - 1)
    //                    = 2 * sum_{p where bit=1} A[i][p] - sum_p A[i][p]
    //                    = 2 * dot(A_row, B_T_col_as_selector) - rowsum
    //
    // For each group of 32 p values, we have one uint32 from B_T[j].
    // We can use this to selectively add A values.
    
    // Precompute row sums
    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;
        
        // Compute rowsum
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        for (size_t p = 0; p < K; p += 4) {
            sum_vec = vaddq_f32(sum_vec, vld1q_f32(a_row + p));
        }
        float rowsum = vaddvq_f32(sum_vec);
        
        for (size_t j = 0; j < K; ++j) {
            const uint32_t* bt_row = B_T + j * K_ints;
            
            // Compute sum of A[i][p] where B_T[j] bit p is set
            float pos_sum = 0.0f;
            
            for (size_t pb = 0; pb < K_ints; ++pb) {
                uint32_t mask = bt_row[pb];
                const float* a_ptr = a_row + pb * 32;
                
                // Sum a_ptr[bit] for each set bit
                while (mask) {
                    int bit = __builtin_ctz(mask);
                    pos_sum += a_ptr[bit];
                    mask &= mask - 1;
                }
            }
            
            c_row[j] = 2.0f * pos_sum - rowsum;
        }
    }
}
