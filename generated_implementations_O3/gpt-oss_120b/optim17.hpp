#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
// NEON intrinsics are available.

// Compile‑time table: for each possible byte value store 8 floats (+1.0f or -1.0f).
constexpr std::array<std::array<float,8>,256> sign_table = [](){
    std::array<std::array<float,8>,256> tbl{};
    for (size_t b = 0; b < 256; ++b){
        for (size_t bit = 0; bit < 8; ++bit){
            tbl[b][bit] = ((b >> bit) & 1u) ? 1.0f : -1.0f;
        }
    }
    return tbl;
}();

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K){
    const size_t K_ints = K / 32; // number of 32‑bit words per row of B
    for (size_t i = 0; i < M; ++i){
        const float* a_row = A + i * K;
        float* c_row = C + i * K;
        for (size_t block = 0; block < K_ints; ++block){
            // eight NEON accumulators, each holds 4 floats (total 32 columns).
            float32x4_t acc0 = vdupq_n_f32(0.0f); // b0 low bits 0‑3
            float32x4_t acc1 = vdupq_n_f32(0.0f); // b0 high bits 4‑7
            float32x4_t acc2 = vdupq_n_f32(0.0f); // b1 low
            float32x4_t acc3 = vdupq_n_f32(0.0f); // b1 high
            float32x4_t acc4 = vdupq_n_f32(0.0f); // b2 low
            float32x4_t acc5 = vdupq_n_f32(0.0f); // b2 high
            float32x4_t acc6 = vdupq_n_f32(0.0f); // b3 low
            float32x4_t acc7 = vdupq_n_f32(0.0f); // b3 high
            for (size_t p = 0; p < K; ++p){
                float32x4_t a_vec = vdupq_n_f32(a_row[p]);
                uint32_t packed = B[p * K_ints + block];
                // extract the four bytes of the packed word
                uint8_t b0 = packed & 0xFFu;
                uint8_t b1 = (packed >> 8) & 0xFFu;
                uint8_t b2 = (packed >> 16) & 0xFFu;
                uint8_t b3 = (packed >> 24) & 0xFFu;
                // load low and high 4‑element sign vectors for each byte
                float32x4_t s0_low  = vld1q_f32(sign_table[b0].data());
                float32x4_t s0_high = vld1q_f32(sign_table[b0].data() + 4);
                float32x4_t s1_low  = vld1q_f32(sign_table[b1].data());
                float32x4_t s1_high = vld1q_f32(sign_table[b1].data() + 4);
                float32x4_t s2_low  = vld1q_f32(sign_table[b2].data());
                float32x4_t s2_high = vld1q_f32(sign_table[b2].data() + 4);
                float32x4_t s3_low  = vld1q_f32(sign_table[b3].data());
                float32x4_t s3_high = vld1q_f32(sign_table[b3].data() + 4);
                // accumulate
                acc0 = vmlaq_f32(acc0, s0_low,  a_vec);
                acc1 = vmlaq_f32(acc1, s0_high, a_vec);
                acc2 = vmlaq_f32(acc2, s1_low,  a_vec);
                acc3 = vmlaq_f32(acc3, s1_high, a_vec);
                acc4 = vmlaq_f32(acc4, s2_low,  a_vec);
                acc5 = vmlaq_f32(acc5, s2_high, a_vec);
                acc6 = vmlaq_f32(acc6, s3_low,  a_vec);
                acc7 = vmlaq_f32(acc7, s3_high, a_vec);
            }
            // store results back to C in correct column order
            size_t base = block * 32;
            vst1q_f32(c_row + base + 0,  acc0); // cols 0‑3
            vst1q_f32(c_row + base + 4,  acc1); // cols 4‑7
            vst1q_f32(c_row + base + 8,  acc2); // cols 8‑11
            vst1q_f32(c_row + base + 12, acc3); // cols 12‑15
            vst1q_f32(c_row + base + 16, acc4); // cols 16‑19
            vst1q_f32(c_row + base + 20, acc5); // cols 20‑23
            vst1q_f32(c_row + base + 24, acc6); // cols 24‑27
            vst1q_f32(c_row + base + 28, acc7); // cols 28‑31
        }
    }
}
