#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
// NEON intrinsics are provided by the harness.

// Compile‑time generation of 4‑float sign vectors for each 4‑bit pattern.
constexpr std::array<std::array<float,4>,16> make_sign4_table() {
    std::array<std::array<float,4>,16> tbl{};
    for (int pat = 0; pat < 16; ++pat) {
        for (int bit = 0; bit < 4; ++bit) {
            tbl[pat][bit] = ((pat >> bit) & 1) ? 1.0f : -1.0f;
        }
    }
    return tbl;
}
constexpr auto sign4_table = make_sign4_table();

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32; // 32‑bit words per row of B
    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;
        // Process each block of 32 columns.
        for (size_t block = 0; block < K_ints; ++block) {
            // Eight NEON accumulators, each holds 4 floats (total 32).
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            float32x4_t acc4 = vdupq_n_f32(0.0f);
            float32x4_t acc5 = vdupq_n_f32(0.0f);
            float32x4_t acc6 = vdupq_n_f32(0.0f);
            float32x4_t acc7 = vdupq_n_f32(0.0f);
            for (size_t p = 0; p < K; ++p) {
                const float a_val = a_row[p];
                const uint32_t packed = B[p * K_ints + block];
                // Extract eight 4‑bit nibbles.
                const uint32_t nib0 = (packed >>  0) & 0xF;
                const uint32_t nib1 = (packed >>  4) & 0xF;
                const uint32_t nib2 = (packed >>  8) & 0xF;
                const uint32_t nib3 = (packed >> 12) & 0xF;
                const uint32_t nib4 = (packed >> 16) & 0xF;
                const uint32_t nib5 = (packed >> 20) & 0xF;
                const uint32_t nib6 = (packed >> 24) & 0xF;
                const uint32_t nib7 = (packed >> 28) & 0xF;
                // Load sign vectors and perform multiply‑add.
                acc0 = vmlaq_n_f32(acc0, vld1q_f32(sign4_table[nib0].data()), a_val);
                acc1 = vmlaq_n_f32(acc1, vld1q_f32(sign4_table[nib1].data()), a_val);
                acc2 = vmlaq_n_f32(acc2, vld1q_f32(sign4_table[nib2].data()), a_val);
                acc3 = vmlaq_n_f32(acc3, vld1q_f32(sign4_table[nib3].data()), a_val);
                acc4 = vmlaq_n_f32(acc4, vld1q_f32(sign4_table[nib4].data()), a_val);
                acc5 = vmlaq_n_f32(acc5, vld1q_f32(sign4_table[nib5].data()), a_val);
                acc6 = vmlaq_n_f32(acc6, vld1q_f32(sign4_table[nib6].data()), a_val);
                acc7 = vmlaq_n_f32(acc7, vld1q_f32(sign4_table[nib7].data()), a_val);
            }
            // Store accumulated results back to C.
            const size_t base = block * 32;
            vst1q_f32(c_row + base +  0, acc0);
            vst1q_f32(c_row + base +  4, acc1);
            vst1q_f32(c_row + base +  8, acc2);
            vst1q_f32(c_row + base + 12, acc3);
            vst1q_f32(c_row + base + 16, acc4);
            vst1q_f32(c_row + base + 20, acc5);
            vst1q_f32(c_row + base + 24, acc6);
            vst1q_f32(c_row + base + 28, acc7);
        }
    }
}
