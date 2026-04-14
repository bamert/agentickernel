#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
// NEON intrinsics are available in the environment.

// Compile‑time table: for each byte value we store 8 floats (+1 or -1).
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
    const size_t K_ints = K / 32; // 32‑bit words per B row
    for (size_t i = 0; i < M; ++i){
        const float* a_row = A + i * K;
        float* c_row = C + i * K;
        for (size_t block = 0; block < K_ints; ++block){
            // eight NEON accumulators, each holds 4 columns (total 32)
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            float32x4_t acc4 = vdupq_n_f32(0.0f);
            float32x4_t acc5 = vdupq_n_f32(0.0f);
            float32x4_t acc6 = vdupq_n_f32(0.0f);
            float32x4_t acc7 = vdupq_n_f32(0.0f);
            for (size_t p = 0; p < K; ++p){
                float32x4_t a_vec = vdupq_n_f32(a_row[p]);
                uint32_t packed = B[p * K_ints + block];
                // extract four bytes
                uint8_t b0 = packed & 0xFFu;
                uint8_t b1 = (packed >> 8) & 0xFFu;
                uint8_t b2 = (packed >> 16) & 0xFFu;
                uint8_t b3 = (packed >> 24) & 0xFFu;
                // load sign vectors (4 floats each) from the table
                float32x4_t s0 = vld1q_f32(sign_table[b0].data());
                float32x4_t s1 = vld1q_f32(sign_table[b1].data());
                float32x4_t s2 = vld1q_f32(sign_table[b2].data());
                float32x4_t s3 = vld1q_f32(sign_table[b3].data());
                // each byte gives 8 columns, we process as two SIMD vectors per byte
                // lower 4 bits -> first vector, higher 4 bits -> second vector
                // sign_table stores 8 consecutive floats; we can use vld1q_f32 for first 4 and vld1q_f32 for next 4 via offset.
                float32x4_t s0_hi = vld1q_f32(sign_table[b0].data() + 4);
                float32x4_t s1_hi = vld1q_f32(sign_table[b1].data() + 4);
                float32x4_t s2_hi = vld1q_f32(sign_table[b2].data() + 4);
                float32x4_t s3_hi = vld1q_f32(sign_table[b3].data() + 4);
                // accumulate
                acc0 = vmlaq_f32(acc0, s0, a_vec);
                acc1 = vmlaq_f32(acc1, s1, a_vec);
                acc2 = vmlaq_f32(acc2, s2, a_vec);
                acc3 = vmlaq_f32(acc3, s3, a_vec);
                acc4 = vmlaq_f32(acc4, s0_hi, a_vec);
                acc5 = vmlaq_f32(acc5, s1_hi, a_vec);
                acc6 = vmlaq_f32(acc6, s2_hi, a_vec);
                acc7 = vmlaq_f32(acc7, s3_hi, a_vec);
            }
            // store back 32 results
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
