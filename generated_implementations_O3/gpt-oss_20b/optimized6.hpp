#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Optimized matrix multiplication using NEON with efficient bit handling.
// This version eliminates the heavy branch inside the inner unrolled loop by
// keeping a shift counter modulo 32 and refilling the packed word when the
// shift wraps. The overall algorithm remains the same as the previous
// fast implementation.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    const size_t block = 8;

    // Zero output matrix C.
    for (size_t idx = 0; idx < M * K; ++idx) {
        C[idx] = 0.0f;
    }

    for (size_t i = 0; i < M; ++i) {
        float* c_row = &C[i * K];
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const uint32_t* b_row = &B[p * K_ints];
            size_t b_index = 0;                // index into B row
            uint32_t packed = b_row[b_index];   // current 32‑bit chunk
            unsigned shift = 0;

            size_t j = 0;
            for (; j + block <= K; j += block) {
                // Load two 4‑lane vectors of the current C block
                float32x4_t c_low = vld1q_f32(&c_row[j]);
                float32x4_t c_high = vld1q_f32(&c_row[j + 4]);

                // Process 8 columns with unrolled loop and updated shift logic
                float32x4_t a_vec = vdupq_n_f32(a_val);

                // Column 0
                unsigned bit0 = (packed >> shift) & 1u;
                float32x4_t s0 = vdupq_n_f32(bit0 ? 1.0f : -1.0f);
                c_low = vaddq_f32(c_low, vmulq_f32(a_vec, s0));
                shift = (shift + 1) & 31u;
                if (shift == 0) { ++b_index; packed = b_row[b_index]; }

                // Column 1
                unsigned bit1 = (packed >> shift) & 1u;
                float32x4_t s1 = vdupq_n_f32(bit1 ? 1.0f : -1.0f);
                c_low = vaddq_f32(c_low, vmulq_f32(a_vec, s1));
                shift = (shift + 1) & 31u;
                if (shift == 0) { ++b_index; packed = b_row[b_index]; }

                // Column 2
                unsigned bit2 = (packed >> shift) & 1u;
                float32x4_t s2 = vdupq_n_f32(bit2 ? 1.0f : -1.0f);
                c_low = vaddq_f32(c_low, vmulq_f32(a_vec, s2));
                shift = (shift + 1) & 31u;
                if (shift == 0) { ++b_index; packed = b_row[b_index]; }

                // Column 3
                unsigned bit3 = (packed >> shift) & 1u;
                float32x4_t s3 = vdupq_n_f32(bit3 ? 1.0f : -1.0f);
                c_low = vaddq_f32(c_low, vmulq_f32(a_vec, s3));
                shift = (shift + 1) & 31u;
                if (shift == 0) { ++b_index; packed = b_row[b_index]; }

                // Column 4
                unsigned bit4 = (packed >> shift) & 1u;
                float32x4_t s4 = vdupq_n_f32(bit4 ? 1.0f : -1.0f);
                c_high = vaddq_f32(c_high, vmulq_f32(a_vec, s4));
                shift = (shift + 1) & 31u;
                if (shift == 0) { ++b_index; packed = b_row[b_index]; }

                // Column 5
                unsigned bit5 = (packed >> shift) & 1u;
                float32x4_t s5 = vdupq_n_f32(bit5 ? 1.0f : -1.0f);
                c_high = vaddq_f32(c_high, vmulq_f32(a_vec, s5));
                shift = (shift + 1) & 31u;
                if (shift == 0) { ++b_index; packed = b_row[b_index]; }

                // Column 6
                unsigned bit6 = (packed >> shift) & 1u;
                float32x4_t s6 = vdupq_n_f32(bit6 ? 1.0f : -1.0f);
                c_high = vaddq_f32(c_high, vmulq_f32(a_vec, s6));
                shift = (shift + 1) & 31u;
                if (shift == 0) { ++b_index; packed = b_row[b_index]; }

                // Column 7
                unsigned bit7 = (packed >> shift) & 1u;
                float32x4_t s7 = vdupq_n_f32(bit7 ? 1.0f : -1.0f);
                c_high = vaddq_f32(c_high, vmulq_f32(a_vec, s7));
                shift = (shift + 1) & 31u;
                if (shift == 0) { ++b_index; packed = b_row[b_index]; }

                // Store results back
                vst1q_f32(&c_row[j], c_low);
                vst1q_f32(&c_row[j + 4], c_high);
            }

            // Tail columns
            for (; j < K; ++j) {
                unsigned bit = (packed >> shift) & 1u;
                float sign = bit ? 1.0f : -1.0f;
                c_row[j] += a_val * sign;

                shift = (shift + 1) & 31u;
                if (shift == 0) { ++b_index; packed = b_row[b_index]; }
            }
        }
    }
}
