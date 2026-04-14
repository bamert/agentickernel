#pragma once
#include <arm_neon.h>

// Basic type definitions for compilation without external headers
using uint32_t = unsigned int;
using size_t   = unsigned long;

/*
 * Matrix multiplication – NEON accelerated variant with 8‑bit sign lookup.
 * The implementation improves the scalar zeroing step with NEON, and
 * performs per-byte accumulation using two 4‑element SIMD vectors.
 */
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    const size_t RowC   = K;

    // Build sign lookup table on the stack (8 floats per byte value)
    alignas(16) float signs_table[256 * 8];
    for (int bv = 0; bv < 256; ++bv) {
        for (int bit = 0; bit < 8; ++bit) {
            signs_table[(bv << 3) | bit] = (bv & (1u << bit)) ? 1.0f : -1.0f;
        }
    }

    const float32x4_t zero_vec = vdupq_n_f32(0.0f);

    for (size_t i = 0; i < M; ++i) {
        const float* Ai = A + i * K;
        float* Ci = C + i * RowC;

        /* Zero the output row using NEON */
        for (size_t j = 0; j < K; j += 4) {
            vst1q_f32(&Ci[j], zero_vec);
        }

        for (size_t p = 0; p < K; ++p) {
            const float a_val = Ai[p];
            const uint32_t* B_row = B + p * K_ints;

            for (size_t w = 0; w < K_ints; ++w) {
                const uint32_t word = B_row[w];
                const size_t base = w * 32;

                for (int byte_offset = 0; byte_offset < 4; ++byte_offset) {
                    uint8_t byte_val = static_cast<uint8_t>(word >> (byte_offset * 8));
                    const float* sign_ptr = &signs_table[(byte_val << 3)];

                    /* Load current C values */
                    float32x4_t c0 = vld1q_f32(&Ci[base + byte_offset * 8 + 0]);
                    float32x4_t c1 = vld1q_f32(&Ci[base + byte_offset * 8 + 4]);
                    /* Load signs */
                    float32x4_t s0 = vld1q_f32(sign_ptr);
                    float32x4_t s1 = vld1q_f32(sign_ptr + 4);
                    /* Broadcast a_val */
                    float32x4_t va = vdupq_n_f32(a_val);
                    /* Accumulate */
                    c0 = vaddq_f32(c0, vmulq_f32(va, s0));
                    c1 = vaddq_f32(c1, vmulq_f32(va, s1));
                    /* Store back */
                    vst1q_f32(&Ci[base + byte_offset * 8 + 0], c0);
                    vst1q_f32(&Ci[base + byte_offset * 8 + 4], c1);
                }
            }
        }
    }
}
