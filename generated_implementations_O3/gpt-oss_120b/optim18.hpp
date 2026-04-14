#pragma once

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <array>

struct SignVec {
    float32x4_t low;  // signs for bits 0..3
    float32x4_t high; // signs for bits 4..7
};

// Compile‑time creation of a lookup table that maps a byte value to two NEON vectors
// containing +1.0f or -1.0f for each of its 8 bits.
constexpr std::array<SignVec,256> make_sign_vec_table() {
    std::array<SignVec,256> tbl{};
    for (size_t b = 0; b < 256; ++b) {
        float low[4];
        float high[4];
        for (int bit = 0; bit < 8; ++bit) {
            float sign = ((b >> bit) & 1u) ? 1.0f : -1.0f;
            if (bit < 4) low[bit] = sign; else high[bit-4] = sign;
        }
        tbl[b].low  = vld1q_f32(low);
        tbl[b].high = vld1q_f32(high);
    }
    return tbl;
}
constexpr auto sign_vec_table = make_sign_vec_table();

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32; // 32‑bit words per row of B
    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;
        for (size_t block = 0; block < K_ints; ++block) {
            // Eight accumulators, each holds 4 columns (total 32).
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            float32x4_t acc4 = vdupq_n_f32(0.0f);
            float32x4_t acc5 = vdupq_n_f32(0.0f);
            float32x4_t acc6 = vdupq_n_f32(0.0f);
            float32x4_t acc7 = vdupq_n_f32(0.0f);
            for (size_t p = 0; p < K; ++p) {
                float32x4_t a_vec = vdupq_n_f32(a_row[p]);
                uint32_t packed = B[p * K_ints + block];
                // extract the four bytes
                uint8_t b0 = packed & 0xFFu;
                uint8_t b1 = (packed >> 8) & 0xFFu;
                uint8_t b2 = (packed >> 16) & 0xFFu;
                uint8_t b3 = (packed >> 24) & 0xFFu;
                const SignVec& sv0 = sign_vec_table[b0];
                const SignVec& sv1 = sign_vec_table[b1];
                const SignVec& sv2 = sign_vec_table[b2];
                const SignVec& sv3 = sign_vec_table[b3];
                // accumulate low and high parts separately
                acc0 = vmlaq_f32(acc0, sv0.low,  a_vec);
                acc1 = vmlaq_f32(acc1, sv0.high, a_vec);
                acc2 = vmlaq_f32(acc2, sv1.low,  a_vec);
                acc3 = vmlaq_f32(acc3, sv1.high, a_vec);
                acc4 = vmlaq_f32(acc4, sv2.low,  a_vec);
                acc5 = vmlaq_f32(acc5, sv2.high, a_vec);
                acc6 = vmlaq_f32(acc6, sv3.low,  a_vec);
                acc7 = vmlaq_f32(acc7, sv3.high, a_vec);
            }
            size_t base = block * 32;
            vst1q_f32(c_row + base + 0,  acc0);
            vst1q_f32(c_row + base + 4,  acc1);
            vst1q_f32(c_row + base + 8,  acc2);
            vst1q_f32(c_row + base + 12, acc3);
            vst1q_f32(c_row + base + 16, acc4);
            vst1q_f32(c_row + base + 20, acc5);
            vst1q_f32(c_row + base + 24, acc6);
            vst1q_f32(c_row + base + 28, acc7);
        }
    }
}
