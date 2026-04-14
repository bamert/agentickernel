#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <arm_neon.h>
#include <array>

// Pre‑computed sign table for 8‑bit masks.
constexpr auto make_sign_table() {
    std::array<std::array<float, 8>, 256> table{};
    for (int m = 0; m < 256; ++m) {
        for (int b = 0; b < 8; ++b) {
            table[m][b] = ((m >> b) & 1U) ? 1.0f : -1.0f;
        }
    }
    return table;
}
constexpr auto sign_table = make_sign_table();

// Matrix multiplication: A (M × K) · B (K × K) → C (M × K)
// Optimized by unrolling the column loop to process 16 columns per
// iteration and using the pre‑computed sign table.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    const size_t block = 16; // 16 columns per iteration

    std::memset(C, 0, M * K * sizeof(float));

    for (size_t i = 0; i < M; ++i) {
        float* c_row = &C[i * K];
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const uint32_t* b_row = &B[p * K_ints];
            size_t b_index = 0;
            uint32_t packed = b_row[b_index];
            unsigned shift = 0;

            size_t j = 0;
            for (; j + block <= K; j += block) {
                // Load C low/high for first 8 columns
                float32x4_t c_low1  = vld1q_f32(&c_row[j]);
                float32x4_t c_high1 = vld1q_f32(&c_row[j + 4]);
                // Load C low/high for next 8 columns
                float32x4_t c_low2  = vld1q_f32(&c_row[j + 8]);
                float32x4_t c_high2 = vld1q_f32(&c_row[j + 12]);

                float32x4_t a_vec = vdupq_n_f32(a_val);

                // First 8‑bit mask
                unsigned mask1 = (packed >> shift) & 0xFFU;
                float32x4_t s_low1  = vld1q_f32(&sign_table[mask1][0]);
                float32x4_t s_high1 = vld1q_f32(&sign_table[mask1][4]);

                // Second 8‑bit mask (possibly crossing 32‑bit boundary)
                unsigned mask2 = (packed >> (shift + 8)) & 0xFFU;
                float32x4_t s_low2  = vld1q_f32(&sign_table[mask2][0]);
                float32x4_t s_high2 = vld1q_f32(&sign_table[mask2][4]);

                // Accumulate first block
                c_low1  = vaddq_f32(c_low1,  vmulq_f32(a_vec, s_low1));
                c_high1 = vaddq_f32(c_high1, vmulq_f32(a_vec, s_high1));
                // Accumulate second block
                c_low2  = vaddq_f32(c_low2,  vmulq_f32(a_vec, s_low2));
                c_high2 = vaddq_f32(c_high2, vmulq_f32(a_vec, s_high2));

                vst1q_f32(&c_row[j], c_low1);
                vst1q_f32(&c_row[j + 4], c_high1);
                vst1q_f32(&c_row[j + 8], c_low2);
                vst1q_f32(&c_row[j + 12], c_high2);

                // Update shift and packed for next 16 columns
                shift += 16;
                while (shift >= 32) {
                    shift -= 32;
                    ++b_index;
                    packed = b_row[b_index];
                }
            }

            // Tail columns
            for (; j < K; ++j) {
                unsigned bit = (packed >> shift) & 1U;
                float sign = bit ? 1.0f : -1.0f;
                c_row[j] += a_val * sign;

                shift = (shift + 1) & 31U;
                if (shift == 0) {
                    ++b_index;
                    packed = b_row[b_index];
                }
            }
        }
    }
}
