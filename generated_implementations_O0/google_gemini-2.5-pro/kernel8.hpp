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
// 1. Simplified SIMD Sign Generation: This kernel replaces the complex and slow
//    sign generation from kernel7 with a much more efficient, direct method.
// 2. High-Throughput Intrinsics:
//    a) `vandq_u32` isolates the bit of interest.
//    b) `vcgtq_u32` (vector compare greater than zero) creates a selection mask.
//       This is a single instruction to turn a non-zero/zero value into an
//       all-ones/all-zeros mask, which is ideal for this task.
//    c) `vbslq_f32` (bitwise select) uses this mask to choose, per-lane,
//       between a vector of all `+1.0f`s and a vector of all `-1.0f`s.
// 3. Reduced Instruction Count: This new method reduces the instruction count
//    inside the critical unrolled loop, lowering overhead and allowing the
//    powerful `vfmaq_f32` instructions to dominate the execution time.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* A_row = A + i * K;
        float* C_row = C + i * K;

        for (size_t j_chunk = 0; j_chunk < K_ints; ++j_chunk) {
            float32x4_t sum_vecs[8] = { vdupq_n_f32(0.0f) };
            
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

            for (size_t p = 0; p < K; ++p) {
                const float a_val = A_row[p];
                const float32x4_t v_a = vdupq_n_f32(a_val);
                const uint32_t packed_b = B[p * K_ints + j_chunk];
                const uint32x4_t v_packed_b = vdupq_n_u32(packed_b);

#define GEN_SIGNS_AND_FMA(idx, pos_mask) do { \
    uint32x4_t isolated = vandq_u32(v_packed_b, pos_mask); \
    uint32x4_t mask = vcgtq_u32(isolated, v_zero); \
    float32x4_t v_signs = vbslq_f32(mask, v_one, v_neg_one); \
    sum_vecs[idx] = vfmaq_f32(sum_vecs[idx], v_a, v_signs); \
} while (0)

                GEN_SIGNS_AND_FMA(0, pos_mask0);
                GEN_SIGNS_AND_FMA(1, pos_mask1);
                GEN_SIGNS_AND_FMA(2, pos_mask2);
                GEN_SIGNS_AND_FMA(3, pos_mask3);
                GEN_SIGNS_AND_FMA(4, pos_mask4);
                GEN_SIGNS_AND_FMA(5, pos_mask5);
                GEN_SIGNS_AND_FMA(6, pos_mask6);
                GEN_SIGNS_AND_FMA(7, pos_mask7);

#undef GEN_SIGNS_AND_FMA
            }
            
            float* C_chunk_ptr = C_row + j_chunk * 32;
            vst1q_f32(C_chunk_ptr + 0, sum_vecs[0]);
            vst1q_f32(C_chunk_ptr + 4, sum_vecs[1]);
            vst1q_f32(C_chunk_ptr + 8, sum_vecs[2]);
            vst1q_f32(C_chunk_ptr + 12, sum_vecs[3]);
            vst1q_f32(C_chunk_ptr + 16, sum_vecs[4]);
            vst1q_f32(C_chunk_ptr + 20, sum_vecs[5]);
            vst1q_f32(C_chunk_ptr + 24, sum_vecs[6]);
            vst1q_f32(C_chunk_ptr + 28, sum_vecs[7]);
        }
    }
}
