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
// 1. SIMD Vectorization (NEON): The scalar logic from kernel3 is promoted to
//    use NEON intrinsics. The 32-element accumulator array is replaced with
//    8 `float32x4_t` vectors, allowing 4 calculations to be done in parallel.
// 2. Vectorized Fused Multiply-Add: The core operation is performed using
//    `vfmaq_f32`, which is highly efficient.
// 3. Branchless Sign Generation: We generate the sign vectors (+1.0f/-1.0f)
//    without branches. This is done by arithmetically manipulating the bits from
//    the packed B matrix to generate the IEEE 754 representation of the signs.
//    Specifically, `0x3f800000` is `1.0f`. We XOR it with `0x80000000` (the sign
//    bit) when the matrix bit is 0. The mask `((bit - 1) & 0x80000000)`
//    achieves this branchlessly.
// 4. Fully Unrolled Inner Loop: The 32 bits from the packed integer are
//    processed in 8 unrolled steps. This eliminates all inner loop overhead
//    and maximizes instruction-level parallelism for the CPU to exploit.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* A_row = A + i * K;
        float* C_row = C + i * K;

        for (size_t j_chunk = 0; j_chunk < K_ints; ++j_chunk) {
            float32x4_t sum_vecs[8] = {
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)
            };

            for (size_t p = 0; p < K; ++p) {
                const float a_val = A_row[p];
                const float32x4_t v_a = vdupq_n_f32(a_val);
                const uint32_t packed_b = B[p * K_ints + j_chunk];
                const uint32_t val_one_f = 0x3f800000;

#define GEN_SIGNS_AND_FMA(chunk_idx, bit_offset) do { \
    uint32_t signs_as_ints[4] = { \
        val_one_f ^ ((((packed_b >> (bit_offset + 0)) & 1) - 1) & 0x80000000), \
        val_one_f ^ ((((packed_b >> (bit_offset + 1)) & 1) - 1) & 0x80000000), \
        val_one_f ^ ((((packed_b >> (bit_offset + 2)) & 1) - 1) & 0x80000000), \
        val_one_f ^ ((((packed_b >> (bit_offset + 3)) & 1) - 1) & 0x80000000)  \
    }; \
    float32x4_t v_signs = vreinterpretq_f32_u32(vld1q_u32(signs_as_ints)); \
    sum_vecs[chunk_idx] = vfmaq_f32(sum_vecs[chunk_idx], v_a, v_signs); \
} while (0)

                GEN_SIGNS_AND_FMA(0, 0);
                GEN_SIGNS_AND_FMA(1, 4);
                GEN_SIGNS_AND_FMA(2, 8);
                GEN_SIGNS_AND_FMA(3, 12);
                GEN_SIGNS_AND_FMA(4, 16);
                GEN_SIGNS_AND_FMA(5, 20);
                GEN_SIGNS_AND_FMA(6, 24);
                GEN_SIGNS_AND_FMA(7, 28);

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
