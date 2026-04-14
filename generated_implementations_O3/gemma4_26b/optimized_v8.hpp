#pragma once
#include <cstdint>
#include <cstddef>

// Optimized Matmul using NEON-friendly loop structure.
// We focus on an i-p-c-k loop order to ensure maximum cache locality.
// We use manual unrolling and bit-masking to reduce branch frequency.
// The goal is to provide a structure that is easily vectorized by the compiler
// using NEON instructions (VLD1, VADD, VBSL).
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* c_row = &C[i * K];
        for (size_t j = 0; j < K; ++j) {
            c_row[j] = 0.0f;
        }

        const float* a_row = &A[i * K];

        for (size_t p = 0; p < K; ++p) {
            const float a_val = a_row[p];
            const float a_neg = -a_val;
            const uint32_t* b_row = &B[p * K_ints];

            for (size_t c = 0; c < K_ints; ++c) {
                const uint32_t packed = b_row[c];
                float* c_chunk = &c_row[c * 32];

                // Processing the 32 bits of the packed integer.
                // We use a pattern that the compiler can likely transform 
                // into a single bitwise mask and a conditional select.
                #pragma unroll
                for (size_t k = 0; k < 32; ++k) {
                    // Using a bitwise comparison that results in 0 or 1.
                    // This is very friendly to CSEL instructions on ARM.
                    float sign = (packed & (1U << k)) ? a_val : a_neg;
                    c_chunk[k] += sign;
                }
            }
        }
    }
}
