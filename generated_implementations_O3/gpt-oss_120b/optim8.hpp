#pragma once

#include <cstddef>
#include <cstdint>

// Pre‑computed sign vectors for each 4‑bit pattern.
// sign4_table[p][i] = (+1.0f) if bit i of pattern p is 1, else -1.0f
constexpr float sign4_table[16][4] = [](){
    float tbl[16][4] = {};
    for (int pat = 0; pat < 16; ++pat) {
        for (int bit = 0; bit < 4; ++bit) {
            tbl[pat][bit] = ((pat >> bit) & 1) ? 1.0f : -1.0f;
        }
    }
    return tbl;
}();

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32; // blocks per row of B (each block = 32 columns)
    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;
        // Process each 32‑column block.
        for (size_t block = 0; block < K_ints; ++block) {
            // 8 NEON vectors holding 4 floats each = 32 columns.
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            float32x4_t acc4 = vdupq_n_f32(0.0f);
            float32x4_t acc5 = vdupq_n_f32(0.0f);
            float32x4_t acc6 = vdupq_n_f32(0.0f);
            float32x4_t acc7 = vdupq_n_f32(0.0f);
            for (size_t p = 0; p < K; ++p) {
                float a_val = a_row[p];
                uint32_t packed = B[p * K_ints + block];
                // Process eight 4‑bit nibbles.
                uint32_t nibble0 = (packed >>  0) & 0xF;
                uint32_t nibble1 = (packed >>  4) & 0xF;
                uint32_t nibble2 = (packed >>  8) & 0xF;
                uint32_t nibble3 = (packed >> 12) & 0xF;
                uint32_t nibble4 = (packed >> 16) & 0xF;
                uint32_t nibble5 = (packed >> 20) & 0xF;
                uint32_t nibble6 = (packed >> 24) & 0xF;
                uint32_t nibble7 = (packed >> 28) & 0xF;
                // Load sign vectors and multiply‑add.
                acc0 = vmlaq_n_f32(acc0, vld1q_f32(sign4_table[nibble0]), a_val);
                acc1 = vmlaq_n_f32(acc1, vld1q_f32(sign4_table[nibble1]), a_val);
                acc2 = vmlaq_n_f32(acc2, vld1q_f32(sign4_table[nibble2]), a_val);
                acc3 = vmlaq_n_f32(acc3, vld1q_f32(sign4_table[nibble3]), a_val);
                acc4 = vmlaq_n_f32(acc4, vld1q_f32(sign4_table[nibble4]), a_val);
                acc5 = vmlaq_n_f32(acc5, vld1q_f32(sign4_table[nibble5]), a_val);
                acc6 = vmlaq_n_f32(acc6, vld1q_f32(sign4_table[nibble6]), a_val);
                acc7 = vmlaq_n_f32(acc7, vld1q_f32(sign4_table[nibble7]), a_val);
            }
            // Store results back to C.
            size_t base = block * 32;
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
