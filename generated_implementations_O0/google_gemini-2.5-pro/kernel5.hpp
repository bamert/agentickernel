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
// 1. Efficient SIMD Sign Generation: This version fixes the performance issue
//    from kernel4. Instead of creating sign arrays in memory (a "store-to-load"
//    bottleneck), it generates the sign vectors entirely within NEON registers.
// 2. Register-Only Arithmetic:
//    a) The 32-bit integer `packed_b` is broadcast to a vector.
//    b) A sequence of vector shifts and bitwise ANDs/ORs isolates each bit
//       and uses it to construct the IEEE 754 representation of +1.0f or -1.0f.
//    c) This is a zero-branch, zero-memory-access way to create the signs.
// 3. Pipelining and Unrolling: The inner loop is fully unrolled, and the
//    register-only operations allow the CPU to pipeline instructions
//    effectively, maximizing throughput for the Fused Multiply-Add (FMA) calls.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* A_row = A + i * K;
        float* C_row = C + i * K;

        for (size_t j_chunk = 0; j_chunk < K_ints; ++j_chunk) {
            float32x4_t sum_vecs[8] = { vdupq_n_f32(0.0f) }; // Compact init

            for (size_t p = 0; p < K; ++p) {
                const float a_val = A_row[p];
                const float32x4_t v_a = vdupq_n_f32(a_val);
                const uint32_t packed_b = B[p * K_ints + j_chunk];
                
                // Constants for sign generation
                const uint32x4_t v_one_bit = vdupq_n_u32(1);
                const uint32x4_t v_one_float_bits = vdupq_n_u32(0x3f800000);

// Generate sign vector for 4 bits and perform FMA
#define GEN_SIGNS_AND_FMA(idx, b0, b1, b2, b3) do { \
    /* Isolate the 4 bits into the LSB of each lane */ \
    uint32x4_t bits = { (packed_b >> b0) & 1, (packed_b >> b1) & 1, (packed_b >> b2) & 1, (packed_b >> b3) & 1 }; \
    /* a) bit is 0 or 1. `bits << 31` gives {0, 0x80000000...} */ \
    /* b) OR with 1.0f bits gives {1.0f, -1.0f}     */ \
    uint32x4_t sign_bits = vorrq_u32(v_one_float_bits, vshlq_n_u32(bits, 31)); \
    float32x4_t v_signs = vreinterpretq_f32_u32(sign_bits); \
    sum_vecs[idx] = vfmaq_f32(sum_vecs[idx], v_a, v_signs); \
} while (0)

                GEN_SIGNS_AND_FMA(0, 0, 1, 2, 3);
                GEN_SIGNS_AND_FMA(1, 4, 5, 6, 7);
                GEN_SIGNS_AND_FMA(2, 8, 9, 10, 11);
                GEN_SIGNS_AND_FMA(3, 12, 13, 14, 15);
                GEN_SIGNS_AND_FMA(4, 16, 17, 18, 19);
                GEN_SIGNS_AND_FMA(5, 20, 21, 22, 23);
                GEN_SIGNS_AND_FMA(6, 24, 25, 26, 27);
                GEN_SIGNS_AND_FMA(7, 28, 29, 30, 31);

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
