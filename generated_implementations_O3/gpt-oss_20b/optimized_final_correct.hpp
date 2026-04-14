#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <arm_neon.h>
#include <array>

// Compile‑time sign table for 8‑bit masks (256 × 8 floats).
constexpr auto sign_table = []{
    std::array<std::array<float, 8>, 256> tbl{};
    for (int m = 0; m < 256; ++m) {
        for (int b = 0; b < 8; ++b) {
            tbl[m][b] = ((m >> b) & 1u) ? 1.0f : -1.0f;
        }
    }
    return tbl;
}();

// Matrix multiplication: A (M×K) · B (K×K) → C (M×K)
// B is packed binary: 1 bit = +1.0f, 0 bit = -1.0f.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;   // 32‑bit words per row of B
    const size_t block  = 8;        // columns per SIMD step

    std::memset(C, 0, M * K * sizeof(float));

    for (size_t i = 0; i < M; ++i) {
        const float* a_row = &A[i * K];
        float*        c_row = &C[i * K];
        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            const uint32_t* b_row = &B[p * K_ints];
            size_t b_idx = 0;
            unsigned shift = 0;
            uint32_t packed = b_row[b_idx];

            size_t j = 0;
            for (; j + block <= K; j += block) {
                unsigned mask   = (packed >> shift) & 0xFFu;
                const float* signs = sign_table[mask].data();

                float32x4_t c_low  = vld1q_f32(&c_row[j]);
                float32x4_t c_high = vld1q_f32(&c_row[j + 4]);

                float32x4_t a_vec = vdupq_n_f32(a_val);

                c_low  = vaddq_f32(c_low,  vmulq_f32(a_vec, vld1q_f32(signs)));
                c_high = vaddq_f32(c_high, vmulq_f32(a_vec, vld1q_f32(signs + 4)));

                vst1q_f32(&c_row[j],     c_low);
                vst1q_f32(&c_row[j + 4], c_high);

                unsigned new_shift = shift + block;
                unsigned inc = new_shift >> 5;
                shift = new_shift & 31u;
                b_idx += inc;
                if (inc) packed = b_row[b_idx];
            }

            for (; j < K; ++j) {
                unsigned bit = ((packed >> shift) & 1u);
                float sign = bit ? 1.0f : -1.0f;
                c_row[j] += a_val * sign;
                ++shift;
                if (shift == 32) {
                    shift = 0;
                    ++b_idx;
                    packed = b_row[b_idx];
                }
            }
        }
    }
}
