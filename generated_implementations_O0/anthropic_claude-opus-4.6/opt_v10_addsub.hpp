
#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    size_t i = 0;
    for (; i + 4 <= M; i += 4) {
        float* C_row0 = C + (i + 0) * K;
        float* C_row1 = C + (i + 1) * K;
        float* C_row2 = C + (i + 2) * K;
        float* C_row3 = C + (i + 3) * K;
        const float* A_row0 = A + (i + 0) * K;
        const float* A_row1 = A + (i + 1) * K;
        const float* A_row2 = A + (i + 2) * K;
        const float* A_row3 = A + (i + 3) * K;

        for (size_t j = 0; j < K; ++j) {
            C_row0[j] = 0.0f;
            C_row1[j] = 0.0f;
            C_row2[j] = 0.0f;
            C_row3[j] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) {
            float a0 = A_row0[p];
            float a1 = A_row1[p];
            float a2 = A_row2[p];
            float a3 = A_row3[p];
            const uint32_t* B_row = B + p * K_ints;

            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t packed = B_row[g];
                float* c0 = C_row0 + g * 32;
                float* c1 = C_row1 + g * 32;
                float* c2 = C_row2 + g * 32;
                float* c3 = C_row3 + g * 32;

                for (int b = 0; b < 32; ++b) {
                    if (packed & (1u << b)) {
                        c0[b] += a0;
                        c1[b] += a1;
                        c2[b] += a2;
                        c3[b] += a3;
                    } else {
                        c0[b] -= a0;
                        c1[b] -= a1;
                        c2[b] -= a2;
                        c3[b] -= a3;
                    }
                }
            }
        }
    }

    for (; i < M; ++i) {
        float* C_row = C + i * K;
        const float* A_row = A + i * K;

        for (size_t j = 0; j < K; ++j) C_row[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            float a_val = A_row[p];
            const uint32_t* B_row = B + p * K_ints;

            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t packed = B_row[g];
                float* c_out = C_row + g * 32;
                for (int b = 0; b < 32; ++b) {
                    if (packed & (1u << b)) {
                        c_out[b] += a_val;
                    } else {
                        c_out[b] -= a_val;
                    }
                }
            }
        }
    }
}
