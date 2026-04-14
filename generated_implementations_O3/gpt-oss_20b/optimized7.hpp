#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>
#include <cstring>

// Matrix multiplication: A (M x K), B packed binary (K x K), C output (M x K)
// Optimized with NEON.  The output matrix C is initialized with memset for speed.
// The inner loop processes 8 columns per iteration using unrolled SIMD operations.
// Bit handling is done by maintaining a shift counter modulo 32 and loading a new 32‑bit word when needed.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    const size_t block = 8; // processed per SIMD step

    // Zero output matrix C using memset (float zero is all bits zero).
    std::memset(C, 0, M * K * sizeof(float));

    for (size_t i = 0; i < M; ++i) {
        float* c_row = &C[i * K];
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const uint32_t* b_row = &B[p * K_ints];
            size_t b_index = 0;                // index into B row
            uint32_t packed = b_row[b_index];   // current 32‑bit chunk
            unsigned shift = 0;                 // current bit offset within packed

            size_t j = 0;
            for (; j + block <= K; j += block) {
                // Load the current C block
                float32x4_t c_low  = vld1q_f32(&c_row[j]);
                float32x4_t c_high = vld1q_f32(&c_row[j + 4]);

                float32x4_t a_vec = vdupq_n_f32(a_val);

                // Unrolled 8 columns
                // Column 0
                unsigned bit0 = (packed >> shift) & 1U;
                float32x4_t s0 = vdupq_n_f32(bit0 ? 1.0f : -1.0f);
                c_low = vaddq_f32(c_low, vmulq_f32(a_vec, s0));
                shift = (shift + 1) & 31U;
                if (shift == 0) { ++b_index; packed = b_row[b_index]; }

                // Column 1
                unsigned bit1 = (packed >> shift) & 1U;
                float32x4_t s1 = vdupq_n_f32(bit1 ? 1.0f : -1.0f);
                c_low = vaddq_f32(c_low, vmulq_f32(a_vec, s1));
                shift = (shift + 1) & 31U;
                if (shift == 0) { ++b_index; packed = b_row[b_index]; }

                // Column 2
                unsigned bit2 = (packed >> shift) & 1U;
                float32x4_t s2 = vdupq_n_f32(bit2 ? 1.0f : -1.0f);
                c_low = vaddq_f32(c_low, vmulq_f32(a_vec, s2));
                shift = (shift + 1) & 31U;
                if (shift == 0) { ++b_index; packed = b_row[b_index]; }

                // Column 3
                unsigned bit3 = (packed >> shift) & 1U;
                float32x4_t s3 = vdupq_n_f32(bit3 ? 1.0f : -1.0f);
                c_low = vaddq_f32(c_low, vmulq_f32(a_vec, s3));
                shift = (shift + 1) & 31U;
                if (shift == 0) { ++b_index; packed = b_row[b_index]; }

                // Column 4
                unsigned bit4 = (packed >> shift) & 1U;
                float32x4_t s4 = vdupq_n_f32(bit4 ? 1.0f : -1.0f);
                c_high = vaddq_f32(c_high, vmulq_f32(a_vec, s4));
                shift = (shift + 1) & 31U;
                if (shift == 0) { ++b_index; packed = b_row[b_index]; }

                // Column 5
                unsigned bit5 = (packed >> shift) & 1U;
                float32x4_t s5 = vdupq_n_f32(bit5 ? 1.0f : -1.0f);
                c_high = vaddq_f32(c_high, vmulq_f32(a_vec, s5));
                shift = (shift + 1) & 31U;
                if (shift == 0) { ++b_index; packed = b_row[b_index]; }

                // Column 6
                unsigned bit6 = (packed >> shift) & 1U;
                float32x4_t s6 = vdupq_n_f32(bit6 ? 1.0f : -1.0f);
                c_high = vaddq_f32(c_high, vmulq_f32(a_vec, s6));
                shift = (shift + 1) & 31U;
                if (shift == 0) { ++b_index; packed = b_row[b_index]; }

                // Column 7
                unsigned bit7 = (packed >> shift) & 1U;
                float32x4_t s7 = vdupq_n_f32(bit7 ? 1.0f : -1.0f);
                c_high = vaddq_f32(c_high, vmulq_f32(a_vec, s7));
                shift = (shift + 1) & 31U;
                if (shift == 0) { ++b_index; packed = b_row[b_index]; }

                // Store results
                vst1q_f32(&c_row[j], c_low);
                vst1q_f32(&c_row[j + 4], c_high);
            }

            // Tail columns (may be 0 if K divisible by 8)
            for (; j < K; ++j) {
                unsigned bit = (packed >> shift) & 1U;
                float sign = bit ? 1.0f : -1.0f;
                c_row[j] += a_val * sign;
                shift = (shift + 1) & 31U;
                if (shift == 0) { ++b_index; packed = b_row[b_index]; }
            }
        }
    }
}
