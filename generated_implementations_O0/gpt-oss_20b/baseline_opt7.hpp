#pragma once
#include <arm_neon.h>

// Basic type definitions for compilation without external headers
using uint32_t = unsigned int;
using size_t   = unsigned long;

/*
 * Matrix multiplication – NEON accelerated variant.
 * Matrix C = Matrix A * Matrix B (Naïve Textbook Method).
 * A : M × K float matrix
 * B : K × K packed binary matrix (1 bit = +1.0f, 0 bit = -1.0f)
 * C : M × K float output matrix
 * K is guaranteed to be a multiple of 32.
 */
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* Ai = A + i * K;
        float* Ci       = C + i * K;

        // Zero the output row
        for (size_t j = 0; j < K; ++j) Ci[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            const float a_val = Ai[p];
            const uint32_t* B_row = B + p * K_ints;
            float32x4_t a_vec = vdupq_n_f32(a_val);

            for (size_t w = 0; w < K_ints; ++w) {
                const uint32_t word = B_row[w];
                const size_t base = w * 32;

                // Process 4 columns at a time
                for (size_t b = 0; b < 32; b += 4) {
                    // Extract 4 bits starting at position b
                    const uint32_t mask = (word >> b) & 0xFu;
                    float signs[4];
                    signs[0] = 2.0f * float((mask >> 0) & 1u) - 1.0f;
                    signs[1] = 2.0f * float((mask >> 1) & 1u) - 1.0f;
                    signs[2] = 2.0f * float((mask >> 2) & 1u) - 1.0f;
                    signs[3] = 2.0f * float((mask >> 3) & 1u) - 1.0f;
                    float32x4_t sign_vec = vld1q_f32(signs);

                    float32x4_t c_vec = vld1q_f32(&Ci[base + b]);
                    float32x4_t prod  = vmulq_f32(a_vec, sign_vec);
                    float32x4_t result = vaddq_f32(c_vec, prod);
                    vst1q_f32(&Ci[base + b], result);
                }
            }
        }
    }
}
