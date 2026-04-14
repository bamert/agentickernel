#pragma once
#include <arm_neon.h>

// Basic type definitions for compilation without external headers
using uint32_t = unsigned int;
using size_t   = unsigned long;

/*
 * Matrix multiplication – NEON accelerated version with an
 * 8‑bit sign lookup table.
 * 
 * The algorithm is identical to baseline_opt9 but the inner
 * accumulation is performed using NEON intrinsics.  The sign
 * table is constructed on the stack at function entry.
 */
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    const size_t RowC   = K;

    // Create the sign lookup table on the stack.
    alignas(16) float signs_table[256 * 8];
    for (int byte_val = 0; byte_val < 256; ++byte_val) {
        for (int bit = 0; bit < 8; ++bit) {
            signs_table[(byte_val << 3) | bit] = (byte_val & (1 << bit)) ? 1.0f : -1.0f;
        }
    }

    for (size_t i = 0; i < M; ++i) {
        const float* Ai = A + i * K;
        float*       Ci = C + i * RowC;

        // Zero the output row
        for (size_t j = 0; j < K; ++j) Ci[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            const float   a_val   = Ai[p];
            const float*  B_row   = B + p * K_ints;

            for (size_t w = 0; w < K_ints; ++w) {
                const uint32_t word = B_row[w];
                const size_t   base = w * 32;

                // Process the four bytes of the 32‑bit word
                for (int byte_offset = 0; byte_offset < 4; ++byte_offset) {
                    uint8_t byte_val = static_cast<uint8_t>(word >> (byte_offset * 8));
                    const float* sign_ptr = &signs_table[(byte_val << 3)];

                    // Load current C values
                    float32x4_t c0 = vld1q_f32(&Ci[base + byte_offset * 8 + 0]);
                    float32x4_t c1 = vld1q_f32(&Ci[base + byte_offset * 8 + 4]);
                    // Load signs
                    float32x4_t s0 = vld1q_f32(sign_ptr);
                    float32x4_t s1 = vld1q_f32(sign_ptr + 4);
                    // Broadcast a_val
                    float32x4_t va = vdupq_n_f32(a_val);
                    // Accumulate
                    c0 = vaddq_f32(c0, vmulq_f32(va, s0));
                    c1 = vaddq_f32(c1, vmulq_f32(va, s1));
                    // Store back
                    vst1q_f32(&Ci[base + byte_offset * 8 + 0], c0);
                    vst1q_f32(&Ci[base + byte_offset * 8 + 4], c1);
                }
            }
        }
    }
}
