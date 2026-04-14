#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Optimized matmul using NEON vector intrinsics over blocks of 8 columns.
// The algorithm keeps the same basic structure as optimized1 but uses
// 128‑bit SIMD operations to accumulate 8 partial results in parallel.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    const size_t block = 8;               // Number of columns processed in one SIMD step

    // Zero the output matrix C. Using a simple loop keeps the code portable.
    for (size_t idx = 0; idx < M * K; ++idx) {
        C[idx] = 0.0f;
    }

    for (size_t i = 0; i < M; ++i) {
        float* c_row = &C[i * K];
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const uint32_t* b_row = &B[p * K_ints];

            // Process columns in blocks of 8
            size_t j = 0;
            for (; j + block <= K; j += block) {
                // Load current values of C for this block
                float32x4_t c_low  = vld1q_f32(&c_row[j]);
                float32x4_t c_high = vld1q_f32(&c_row[j + 4]);

                // Extract 8 bits from packed data
                uint32_t packed = b_row[j / 32];
                unsigned shift = j % 32;
                // Build sign array of 8 floats (+1.0f or -1.0f)
                float signs[8];
                for (int t = 0; t < 8; ++t) {
                    uint32_t bit = (packed >> (shift + t)) & 1u;
                    signs[t] = bit ? 1.0f : -1.0f;
                }
                // Load signs into NEON vectors
                float32x4_t s_low  = vld1q_f32(&signs[0]);
                float32x4_t s_high = vld1q_f32(&signs[4]);

                // Broadcast a_val and multiply
                float32x4_t a_vec_low  = vdupq_n_f32(a_val);
                float32x4_t a_vec_high = vdupq_n_f32(a_val);

                c_low  = vaddq_f32(c_low,  vmulq_f32(a_vec_low,  s_low));
                c_high = vaddq_f32(c_high, vmulq_f32(a_vec_high, s_high));

                // Store back
                vst1q_f32(&c_row[j],   c_low);
                vst1q_f32(&c_row[j+4], c_high);
            }

            // Tail columns that do not fill a full block of 8
            for (; j < K; ++j) {
                uint32_t packed = b_row[j / 32];
                unsigned shift = j % 32;
                uint32_t bit = (packed >> shift) & 1u;
                float sign = bit ? 1.0f : -1.0f;
                c_row[j] += a_val * sign;
            }
        }
    }
}
