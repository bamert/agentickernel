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
// 1. Tiling/Blocking (MR=2): The main bottleneck in previous NEON attempts was
//    the high overhead of generating sign vectors relative to the FMA work.
//    This version computes TWO rows of the output matrix C at a time.
// 2. Amortized Sign Generation: By processing two rows, we can generate the
//    eight sign vectors from the B matrix value ONCE and REUSE them for both
//    rows. This doubles the ratio of arithmetic (FMA) to overhead instructions.
// 3. Increased Register Use: This approach uses ~30 NEON registers for
//    accumulators (16), A-values (2), and sign generation, which fits within
//    the available hardware registers, avoiding spills to the stack.
// 4. Cleanup Loop: A simple loop handles the case where M is odd by processing
//    the final row using the previous single-row (kernel8) logic.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    size_t M_main = M - (M % 2);

    const uint32x4_t pos_mask0 = {1U<<0, 1U<<1, 1U<<2, 1U<<3};
    const uint32x4_t pos_mask1 = {1U<<4, 1U<<5, 1U<<6, 1U<<7};
    const uint32x4_t pos_mask2 = {1U<<8, 1U<<9, 1U<<10, 1U<<11};
    const uint32x4_t pos_mask3 = {1U<<12, 1U<<13, 1U<<14, 1U<<15};
    const uint32x4_t pos_mask4 = {1U<<16, 1U<<17, 1U<<18, 1U<<19};
    const uint32x4_t pos_mask5 = {1U<<20, 1U<<21, 1U<<22, 1U<<23};
    const uint32x4_t pos_mask6 = {1U<<24, 1U<<25, 1U<<26, 1U<<27};
    const uint32x4_t pos_mask7 = {1U<<28, 1U<<29, 1U<<30, 1U<<31};
    const float32x4_t v_one = vdupq_n_f32(1.0f);
    const float32x4_t v_neg_one = vdupq_n_f32(-1.0f);
    const uint32x4_t v_zero = vdupq_n_u32(0);

    for (size_t i = 0; i < M_main; i += 2) {
        const float* A_row0 = A + i * K;
        const float* A_row1 = A + (i + 1) * K;
        float* C_row0 = C + i * K;
        float* C_row1 = C + (i + 1) * K;

        for (size_t j_chunk = 0; j_chunk < K_ints; ++j_chunk) {
            float32x4_t sums0[8] = {vdupq_n_f32(0.0f)};
            float32x4_t sums1[8] = {vdupq_n_f32(0.0f)};

            for (size_t p = 0; p < K; ++p) {
                const float32x4_t v_a0 = vdupq_n_f32(A_row0[p]);
                const float32x4_t v_a1 = vdupq_n_f32(A_row1[p]);
                const uint32x4_t v_packed_b = vdupq_n_u32(B[p * K_ints + j_chunk]);

#define GEN_SIGNS_AND_FMA(idx, pos_mask) do { \
    uint32x4_t isolated = vandq_u32(v_packed_b, pos_mask); \
    uint32x4_t mask = vcgtq_u32(isolated, v_zero); \
    float32x4_t v_signs = vbslq_f32(mask, v_one, v_neg_one); \
    sums0[idx] = vfmaq_f32(sums0[idx], v_a0, v_signs); \
    sums1[idx] = vfmaq_f32(sums1[idx], v_a1, v_signs); \
} while (0)

                GEN_SIGNS_AND_FMA(0, pos_mask0); GEN_SIGNS_AND_FMA(1, pos_mask1);
                GEN_SIGNS_AND_FMA(2, pos_mask2); GEN_SIGNS_AND_FMA(3, pos_mask3);
                GEN_SIGNS_AND_FMA(4, pos_mask4); GEN_SIGNS_AND_FMA(5, pos_mask5);
                GEN_SIGNS_AND_FMA(6, pos_mask6); GEN_SIGNS_AND_FMA(7, pos_mask7);
            }
            
            float* C_chunk_ptr0 = C_row0 + j_chunk * 32;
            float* C_chunk_ptr1 = C_row1 + j_chunk * 32;
            for(int j=0; j<8; ++j) {
                vst1q_f32(C_chunk_ptr0 + j*4, sums0[j]);
                vst1q_f32(C_chunk_ptr1 + j*4, sums1[j]);
            }
        }
    }
    
    // Cleanup for odd M
    if (M % 2) {
        size_t i = M - 1;
        const float* A_row = A + i * K;
        float* C_row = C + i * K;
        for (size_t j_chunk = 0; j_chunk < K_ints; ++j_chunk) {
            float32x4_t sums[8] = {vdupq_n_f32(0.0f)};
            for (size_t p = 0; p < K; ++p) {
                const float32x4_t v_a = vdupq_n_f32(A_row[p]);
                const uint32x4_t v_packed_b = vdupq_n_u32(B[p * K_ints + j_chunk]);
#define GEN_SIGNS_AND_FMA_SCALAR(idx, pos_mask) do { \
    uint32x4_t isolated = vandq_u32(v_packed_b, pos_mask); \
    uint32x4_t mask = vcgtq_u32(isolated, v_zero); \
    float32x4_t v_signs = vbslq_f32(mask, v_one, v_neg_one); \
    sums[idx] = vfmaq_f32(sums[idx], v_a, v_signs); \
} while (0)
                GEN_SIGNS_AND_FMA_SCALAR(0, pos_mask0); GEN_SIGNS_AND_FMA_SCALAR(1, pos_mask1);
                GEN_SIGNS_AND_FMA_SCALAR(2, pos_mask2); GEN_SIGNS_AND_FMA_SCALAR(3, pos_mask3);
                GEN_SIGNS_AND_FMA_SCALAR(4, pos_mask4); GEN_SIGNS_AND_FMA_SCALAR(5, pos_mask5);
                GEN_SIGNS_AND_FMA_SCALAR(6, pos_mask6); GEN_SIGNS_AND_FMA_SCALAR(7, pos_mask7);
            }
            float* C_chunk_ptr = C_row + j_chunk * 32;
            for(int j=0; j<8; ++j) vst1q_f32(C_chunk_ptr + j*4, sums[j]);
        }
    }
}
