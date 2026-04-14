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
// 1. Correct SIMD Sign Generation: This fixes the correctness bug and the
//    memory bottleneck from kernel5. The signs are generated entirely in
//    registers using a novel, branchless bit manipulation technique.
// 2. Efficient Vector Logic:
//    a) Constant bit-position masks are created (e.g., {1<<0, 1<<1, ...}).
//    b) The packed integer from B is broadcast to a vector.
//    c) A vector AND isolates the bits.
//    d) A vector compare-equal-to-zero (`vceqzq_u32`) branchlessly generates
//       a mask of all-ones (if bit is 0) or all-zeros (if bit is 1). This is
//       equivalent to `(bit - 1)` from the scalar version.
//    e) This mask is then used to flip the sign bit of a `1.0f` vector,
//       creating the final sign vector without any memory access.
// 3. Full Unrolling & Pipelining: The inner loop is completely unrolled, and
//    the register-only operations enable optimal instruction pipelining.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* A_row = A + i * K;
        float* C_row = C + i * K;

        for (size_t j_chunk = 0; j_chunk < K_ints; ++j_chunk) {
            float32x4_t sum_vecs[8] = { vdupq_n_f32(0.0f) };
            
            // Pre-calculate bit position masks
            const uint32x4_t pos_mask0 = {1<<0, 1<<1, 1<<2, 1<<3};
            const uint32x4_t pos_mask1 = {1<<4, 1<<5, 1<<6, 1<<7};
            const uint32x4_t pos_mask2 = {1<<8, 1<<9, 1<<10, 1<<11};
            const uint32x4_t pos_mask3 = {1<<12, 1<<13, 1<<14, 1<<15};
            const uint32x4_t pos_mask4 = {1<<16, 1<<17, 1<<18, 1<<19};
            const uint32x4_t pos_mask5 = {1<<20, 1<<21, 1<<22, 1<<23};
            const uint32x4_t pos_mask6 = {1<<24, 1<<25, 1<<26, 1<<27};
            const uint32x4_t pos_mask7 = {1<<28, 1<<29, 1<<30, 1<<31};

            for (size_t p = 0; p < K; ++p) {
                const float a_val = A_row[p];
                const float32x4_t v_a = vdupq_n_f32(a_val);
                const uint32_t packed_b = B[p * K_ints + j_chunk];
                const uint32x4_t v_packed_b = vdupq_n_u32(packed_b);

                const uint32x4_t v_sign_bit_mask = vdupq_n_u32(0x80000000);
                const uint32x4_t v_one_float_bits = vdupq_n_u32(0x3f800000);

#define GEN_SIGNS_AND_FMA(idx, pos_mask) do { \
    /* Isolate the bits at their positions */ \
    uint32x4_t isolated = vandq_u32(v_packed_b, pos_mask); \
    /* If bit is 0, isolated is 0. vceqq gives 0xFFFFFFFF (-1). */ \
    /* If bit is 1, isolated is non-zero. vceqq gives 0. */ \
    uint32x4_t bit_minus_1 = vceqzq_u32(isolated); \
    /* Get the sign bit mask (0x80000000 or 0) */ \
    uint32x4_t final_sign_mask = vandq_u32(bit_minus_1, v_sign_bit_mask); \
    /* XOR with 1.0f to flip the sign if needed */ \
    uint32x4_t sign_float_bits = vxorq_u32(final_sign_mask, v_one_float_bits); \
    float32x4_t v_signs = vreinterpretq_f32_u32(sign_float_bits); \
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
