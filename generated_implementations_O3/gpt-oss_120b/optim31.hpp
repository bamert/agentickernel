#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <arm_neon.h>

// Compile‑time sign table: for each possible byte value store 8 floats (+1.0f or -1.0f).
constexpr std::array<std::array<float, 8>, 256> sign_table = [](){
    std::array<std::array<float, 8>, 256> tbl{};
    for (size_t b = 0; b < 256; ++b) {
        for (size_t bit = 0; bit < 8; ++bit) {
            tbl[b][bit] = ((b >> bit) & 1u) ? 1.0f : -1.0f;
        }
    }
    return tbl;
}();

// Matrix multiply using NEON SIMD and pointer‑incremented access to B.
// A: M×K float (row‑major)
// B: K×K packed binary (1 → +1.0f, 0 → -1.0f) stored as K/32 uint32_t per row
// C: M×K float (row‑major)
// K is a multiple of 32.
void matmul(const float* __restrict A,
            const uint32_t* __restrict B,
            float* __restrict C,
            size_t M,
            size_t K)
{
    const size_t K_ints = K / 32; // number of uint32_t words per row of B

    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;
        for (size_t block = 0; block < K_ints; ++block) {
            // eight NEON accumulators (4 floats each) → 32 columns
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            float32x4_t acc4 = vdupq_n_f32(0.0f);
            float32x4_t acc5 = vdupq_n_f32(0.0f);
            float32x4_t acc6 = vdupq_n_f32(0.0f);
            float32x4_t acc7 = vdupq_n_f32(0.0f);

            const uint32_t* b_ptr = B + block; // points to B[0][block]
            // Unroll inner K loop by 4 to reduce overhead.
            for (size_t p = 0; p + 3 < K; p += 4) {
                // ---- element p ----
                float32x4_t a_vec0 = vdupq_n_f32(a_row[p]);
                uint32_t packed0 = *b_ptr; b_ptr += K_ints;
                uint8_t b0_0 = packed0 & 0xFFu;
                uint8_t b0_1 = (packed0 >> 8)  & 0xFFu;
                uint8_t b0_2 = (packed0 >> 16) & 0xFFu;
                uint8_t b0_3 = (packed0 >> 24) & 0xFFu;
                float32x4_t s0_low  = vld1q_f32(sign_table[b0_0].data());
                float32x4_t s0_high = vld1q_f32(sign_table[b0_0].data() + 4);
                float32x4_t s1_low  = vld1q_f32(sign_table[b0_1].data());
                float32x4_t s1_high = vld1q_f32(sign_table[b0_1].data() + 4);
                float32x4_t s2_low  = vld1q_f32(sign_table[b0_2].data());
                float32x4_t s2_high = vld1q_f32(sign_table[b0_2].data() + 4);
                float32x4_t s3_low  = vld1q_f32(sign_table[b0_3].data());
                float32x4_t s3_high = vld1q_f32(sign_table[b0_3].data() + 4);
                acc0 = vmlaq_f32(acc0, s0_low,  a_vec0);
                acc1 = vmlaq_f32(acc1, s0_high, a_vec0);
                acc2 = vmlaq_f32(acc2, s1_low,  a_vec0);
                acc3 = vmlaq_f32(acc3, s1_high, a_vec0);
                acc4 = vmlaq_f32(acc4, s2_low,  a_vec0);
                acc5 = vmlaq_f32(acc5, s2_high, a_vec0);
                acc6 = vmlaq_f32(acc6, s3_low,  a_vec0);
                acc7 = vmlaq_f32(acc7, s3_high, a_vec0);

                // ---- element p+1 ----
                float32x4_t a_vec1 = vdupq_n_f32(a_row[p+1]);
                uint32_t packed1 = *b_ptr; b_ptr += K_ints;
                uint8_t b1_0 = packed1 & 0xFFu;
                uint8_t b1_1 = (packed1 >> 8)  & 0xFFu;
                uint8_t b1_2 = (packed1 >> 16) & 0xFFu;
                uint8_t b1_3 = (packed1 >> 24) & 0xFFu;
                float32x4_t t0_low  = vld1q_f32(sign_table[b1_0].data());
                float32x4_t t0_high = vld1q_f32(sign_table[b1_0].data() + 4);
                float32x4_t t1_low  = vld1q_f32(sign_table[b1_1].data());
                float32x4_t t1_high = vld1q_f32(sign_table[b1_1].data() + 4);
                float32x4_t t2_low  = vld1q_f32(sign_table[b1_2].data());
                float32x4_t t2_high = vld1q_f32(sign_table[b1_2].data() + 4);
                float32x4_t t3_low  = vld1q_f32(sign_table[b1_3].data());
                float32x4_t t3_high = vld1q_f32(sign_table[b1_3].data() + 4);
                acc0 = vmlaq_f32(acc0, t0_low,  a_vec1);
                acc1 = vmlaq_f32(acc1, t0_high, a_vec1);
                acc2 = vmlaq_f32(acc2, t1_low,  a_vec1);
                acc3 = vmlaq_f32(acc3, t1_high, a_vec1);
                acc4 = vmlaq_f32(acc4, t2_low,  a_vec1);
                acc5 = vmlaq_f32(acc5, t2_high, a_vec1);
                acc6 = vmlaq_f32(acc6, t3_low,  a_vec1);
                acc7 = vmlaq_f32(acc7, t3_high, a_vec1);

                // ---- element p+2 ----
                float32x4_t a_vec2 = vdupq_n_f32(a_row[p+2]);
                uint32_t packed2 = *b_ptr; b_ptr += K_ints;
                uint8_t b2_0 = packed2 & 0xFFu;
                uint8_t b2_1 = (packed2 >> 8)  & 0xFFu;
                uint8_t b2_2 = (packed2 >> 16) & 0xFFu;
                uint8_t b2_3 = (packed2 >> 24) & 0xFFu;
                float32x4_t u0_low  = vld1q_f32(sign_table[b2_0].data());
                float32x4_t u0_high = vld1q_f32(sign_table[b2_0].data() + 4);
                float32x4_t u1_low  = vld1q_f32(sign_table[b2_1].data());
                float32x4_t u1_high = vld1q_f32(sign_table[b2_1].data() + 4);
                float32x4_t u2_low  = vld1q_f32(sign_table[b2_2].data());
                float32x4_t u2_high = vld1q_f32(sign_table[b2_2].data() + 4);
                float32x4_t u3_low  = vld1q_f32(sign_table[b2_3].data());
                float32x4_t u3_high = vld1q_f32(sign_table[b2_3].data() + 4);
                acc0 = vmlaq_f32(acc0, u0_low,  a_vec2);
                acc1 = vmlaq_f32(acc1, u0_high, a_vec2);
                acc2 = vmlaq_f32(acc2, u1_low,  a_vec2);
                acc3 = vmlaq_f32(acc3, u1_high, a_vec2);
                acc4 = vmlaq_f32(acc4, u2_low,  a_vec2);
                acc5 = vmlaq_f32(acc5, u2_high, a_vec2);
                acc6 = vmlaq_f32(acc6, u3_low,  a_vec2);
                acc7 = vmlaq_f32(acc7, u3_high, a_vec2);

                // ---- element p+3 ----
                float32x4_t a_vec3 = vdupq_n_f32(a_row[p+3]);
                uint32_t packed3 = *b_ptr; b_ptr += K_ints;
                uint8_t b3_0 = packed3 & 0xFFu;
                uint8_t b3_1 = (packed3 >> 8)  & 0xFFu;
                uint8_t b3_2 = (packed3 >> 16) & 0xFFu;
                uint8_t b3_3 = (packed3 >> 24) & 0xFFu;
                float32x4_t v0_low  = vld1q_f32(sign_table[b3_0].data());
                float32x4_t v0_high = vld1q_f32(sign_table[b3_0].data() + 4);
                float32x4_t v1_low  = vld1q_f32(sign_table[b3_1].data());
                float32x4_t v1_high = vld1q_f32(sign_table[b3_1].data() + 4);
                float32x4_t v2_low  = vld1q_f32(sign_table[b3_2].data());
                float32x4_t v2_high = vld1q_f32(sign_table[b3_2].data() + 4);
                float32x4_t v3_low  = vld1q_f32(sign_table[b3_3].data());
                float32x4_t v3_high = vld1q_f32(sign_table[b3_3].data() + 4);
                acc0 = vmlaq_f32(acc0, v0_low,  a_vec3);
                acc1 = vmlaq_f32(acc1, v0_high, a_vec3);
                acc2 = vmlaq_f32(acc2, v1_low,  a_vec3);
                acc3 = vmlaq_f32(acc3, v1_high, a_vec3);
                acc4 = vmlaq_f32(acc4, v2_low,  a_vec3);
                acc5 = vmlaq_f32(acc5, v2_high, a_vec3);
                acc6 = vmlaq_f32(acc6, v3_low,  a_vec3);
                acc7 = vmlaq_f32(acc7, v3_high, a_vec3);
            }
            // Process any remaining K elements (shouldn't occur for multiples of 32).
            for (size_t p = K - (K % 4); p < K; ++p) {
                float32x4_t a_vec = vdupq_n_f32(a_row[p]);
                uint32_t packed = *b_ptr; b_ptr += K_ints;
                uint8_t b0 = packed & 0xFFu;
                uint8_t b1 = (packed >> 8)  & 0xFFu;
                uint8_t b2 = (packed >> 16) & 0xFFu;
                uint8_t b3 = (packed >> 24) & 0xFFu;
                float32x4_t s0_low  = vld1q_f32(sign_table[b0].data());
                float32x4_t s0_high = vld1q_f32(sign_table[b0].data() + 4);
                float32x4_t s1_low  = vld1q_f32(sign_table[b1].data());
                float32x4_t s1_high = vld1q_f32(sign_table[b1].data() + 4);
                float32x4_t s2_low  = vld1q_f32(sign_table[b2].data());
                float32x4_t s2_high = vld1q_f32(sign_table[b2].data() + 4);
                float32x4_t s3_low  = vld1q_f32(sign_table[b3].data());
                float32x4_t s3_high = vld1q_f32(sign_table[b3].data() + 4);
                acc0 = vmlaq_f32(acc0, s0_low,  a_vec);
                acc1 = vmlaq_f32(acc1, s0_high, a_vec);
                acc2 = vmlaq_f32(acc2, s1_low,  a_vec);
                acc3 = vmlaq_f32(acc3, s1_high, a_vec);
                acc4 = vmlaq_f32(acc4, s2_low,  a_vec);
                acc5 = vmlaq_f32(acc5, s2_high, a_vec);
                acc6 = vmlaq_f32(acc6, s3_low,  a_vec);
                acc7 = vmlaq_f32(acc7, s3_high, a_vec);
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
