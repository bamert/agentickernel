#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <arm_neon.h>

// Precomputed table: for each 8‑bit mask, 8 floats (+1.0f or -1.0f)
constexpr float sign_table[256][8] {
    // Each row initialized manually by a small helper in the compiler?
    // We rely on the compiler to generate the data.
    // The initializer is omitted for brevity.
};

// NOTE: The above initializer requires explicit values; generating all 256 rows
// manually is verbose. Instead we will generate it programmatically at compile time
// using a constexpr lambda.

constexpr float build_sign(int mask, int bit) {
    return ((mask >> bit) & 1) ? 1.0f : -1.0f;
}

constexpr void fill_sign_table(float (&table)[256][8]) {
    for (int m = 0; m < 256; ++m) {
        for (int b = 0; b < 8; ++b) {
            table[m][b] = build_sign(m, b);
        }
    }
}

constexpr float sign_tbl[256][8] = []{
    float tbl[256][8];
    fill_sign_table(tbl);
    return tbl;
}();

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    const size_t block = 8;

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
                float32x4_t c_low  = vld1q_f32(&c_row[j]);
                float32x4_t c_high = vld1q_f32(&c_row[j + 4]);
                float32x4_t a_vec = vdupq_n_f32(a_val);

                uint32_t mask = (packed >> shift) & 0xFFU;
                const float* signs = sign_tbl[mask];
                float32x4_t s_low  = vld1q_f32(signs);
                float32x4_t s_high = vld1q_f32(signs + 4);

                c_low  = vaddq_f32(c_low,  vmulq_f32(a_vec, s_low));
                c_high = vaddq_f32(c_high, vmulq_f32(a_vec, s_high));

                vst1q_f32(&c_row[j], c_low);
                vst1q_f32(&c_row[j + 4], c_high);

                shift += 8;
                if (shift >= 32) {
                    shift -= 32;
                    ++b_index;
                    packed = b_row[b_index];
                }
            }

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
