#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Calculates Matrix C = Matrix A * Matrix B
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
// ---------------------------------------------------------
// Optimization Strategy:
// 1. Algorithmic Transformation (Popcount Sum): This kernel uses a completely
//    different algorithm derived from a mathematical rearrangement of the problem:
//    C[i][j] = 2 * sum(A[i][p] where bit is 1) - sum(all A[i][p]).
// 2. Two-Phase Calculation:
//    a) For each row of A, a single scalar pass calculates the total sum.
//    b) The main NEON loop calculates the "masked sum" by generating a 0/1 mask
//       from the bits in B and uses it to selectively add A's values.
// 3. Different Instruction Mix: This approach avoids the FMA-based computation of
//    previous kernels, instead relying on a sequence of bitwise selections and
//    additions. This represents the final unexplored optimization path and may
//    interact more favorably with the CPU's execution units.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    const uint32x4_t pos_mask0={1U<<0,1U<<1,1U<<2,1U<<3}, pos_mask1={1U<<4,1U<<5,1U<<6,1U<<7},
                     pos_mask2={1U<<8,1U<<9,1U<<10,1U<<11}, pos_mask3={1U<<12,1U<<13,1U<<14,1U<<15},
                     pos_mask4={1U<<16,1U<<17,1U<<18,1U<<19}, pos_mask5={1U<<20,1U<<21,1U<<22,1U<<23},
                     pos_mask6={1U<<24,1U<<25,1U<<26,1U<<27}, pos_mask7={1U<<28,1U<<29,1U<<30,1U<<31};
    const uint32x4_t v_zero_u = vdupq_n_u32(0);
    const float32x4_t v_zero_f = vdupq_n_f32(0.0f);

    for (size_t i = 0; i < M; ++i) {
        const float* A_row = A + i * K;
        float* C_row = C + i * K;

        // Phase 1: Calculate the total sum of the row in A.
        float total_sum_A = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            total_sum_A += A_row[p];
        }
        const float32x4_t v_total_sum_A = vdupq_n_f32(total_sum_A);

        for (size_t j_chunk = 0; j_chunk < K_ints; ++j_chunk) {
            float32x4_t popcount_sums[8] = {v_zero_f};

            // Phase 2: Calculate the masked sum
            for (size_t p = 0; p < K; ++p) {
                const float32x4_t v_a = vdupq_n_f32(A_row[p]);
                const uint32x4_t v_packed_b = vdupq_n_u32(B[p * K_ints + j_chunk]);
                
#define MASKED_ADD(idx, pos_mask) do { \
    uint32x4_t isolated = vandq_u32(v_packed_b, pos_mask); \
    uint32x4_t mask = vcgtq_u32(isolated, v_zero_u); \
    float32x4_t selected_a = vbslq_f32(mask, v_a, v_zero_f); \
    popcount_sums[idx] = vaddq_f32(popcount_sums[idx], selected_a); \
} while (0)
                MASKED_ADD(0, pos_mask0); MASKED_ADD(1, pos_mask1);
                MASKED_ADD(2, pos_mask2); MASKED_ADD(3, pos_mask3);
                MASKED_ADD(4, pos_mask4); MASKED_ADD(5, pos_mask5);
                MASKED_ADD(6, pos_mask6); MASKED_ADD(7, pos_mask7);
            }
            
            // Final calculation: 2 * popcount_sum - total_sum
            float* C_ptr = C_row + j_chunk * 32;
            vst1q_f32(C_ptr + 0,  vsubq_f32(vmulq_n_f32(popcount_sums[0], 2.0f), v_total_sum_A));
            vst1q_f32(C_ptr + 4,  vsubq_f32(vmulq_n_f32(popcount_sums[1], 2.0f), v_total_sum_A));
            vst1q_f32(C_ptr + 8,  vsubq_f32(vmulq_n_f32(popcount_sums[2], 2.0f), v_total_sum_A));
            vst1q_f32(C_ptr + 12, vsubq_f32(vmulq_n_f32(popcount_sums[3], 2.0f), v_total_sum_A));
            vst1q_f32(C_ptr + 16, vsubq_f32(vmulq_n_f32(popcount_sums[4], 2.0f), v_total_sum_A));
            vst1q_f32(C_ptr + 20, vsubq_f32(vmulq_n_f32(popcount_sums[5], 2.0f), v_total_sum_A));
            vst1q_f32(C_ptr + 24, vsubq_f32(vmulq_n_f32(popcount_sums[6], 2.0f), v_total_sum_A));
            vst1q_f32(C_ptr + 28, vsubq_f32(vmulq_n_f32(popcount_sums[7], 2.0f), v_total_sum_A));
        }
    }
}
