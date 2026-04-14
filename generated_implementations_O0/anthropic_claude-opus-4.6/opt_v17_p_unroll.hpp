
#pragma once
#include <cstdint>
#include <cstddef>
#include <string.h>

// Key insight: the bottleneck is load/store of C rows (4 rows × 3072 floats = 48KB)
// per iteration of p. We iterate K=3072 times through p.
// Total C traffic: 3072 × 48KB × 2 (read+write) = ~300MB just for C.
//
// To reduce C traffic, process multiple p values per C sweep.
// For each group of columns (g), load C once, process P_TILE values of p, store C.

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, 
            float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    size_t i = 0;
    for (; i + 4 <= M; i += 4) {
        float* __restrict__ C_row0 = C + (i + 0) * K;
        float* __restrict__ C_row1 = C + (i + 1) * K;
        float* __restrict__ C_row2 = C + (i + 2) * K;
        float* __restrict__ C_row3 = C + (i + 3) * K;
        const float* __restrict__ A_row0 = A + (i + 0) * K;
        const float* __restrict__ A_row1 = A + (i + 1) * K;
        const float* __restrict__ A_row2 = A + (i + 2) * K;
        const float* __restrict__ A_row3 = A + (i + 3) * K;

        memset(C_row0, 0, K * sizeof(float));
        memset(C_row1, 0, K * sizeof(float));
        memset(C_row2, 0, K * sizeof(float));
        memset(C_row3, 0, K * sizeof(float));

        // Process 2 p values at a time to amortize C load/stores
        size_t p = 0;
        for (; p + 2 <= K; p += 2) {
            float a00 = A_row0[p], a01 = A_row0[p+1];
            float a10 = A_row1[p], a11 = A_row1[p+1];
            float a20 = A_row2[p], a21 = A_row2[p+1];
            float a30 = A_row3[p], a31 = A_row3[p+1];
            const uint32_t* B_row0 = B + p * K_ints;
            const uint32_t* B_row1 = B + (p+1) * K_ints;

            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t packed0 = B_row0[g];
                uint32_t packed1 = B_row1[g];
                float* c0 = C_row0 + g * 32;
                float* c1 = C_row1 + g * 32;
                float* c2 = C_row2 + g * 32;
                float* c3 = C_row3 + g * 32;

                for (int b = 0; b < 32; ++b) {
                    float sign0 = (packed0 & (1u << b)) ? 1.0f : -1.0f;
                    float sign1 = (packed1 & (1u << b)) ? 1.0f : -1.0f;
                    float contrib0 = a00 * sign0 + a01 * sign1;
                    float contrib1 = a10 * sign0 + a11 * sign1;
                    float contrib2 = a20 * sign0 + a21 * sign1;
                    float contrib3 = a30 * sign0 + a31 * sign1;
                    c0[b] += contrib0;
                    c1[b] += contrib1;
                    c2[b] += contrib2;
                    c3[b] += contrib3;
                }
            }
        }
        for (; p < K; ++p) {
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
                    float sign = (packed & (1u << b)) ? 1.0f : -1.0f;
                    c0[b] += a0 * sign;
                    c1[b] += a1 * sign;
                    c2[b] += a2 * sign;
                    c3[b] += a3 * sign;
                }
            }
        }
    }

    for (; i < M; ++i) {
        float* C_row = C + i * K;
        const float* A_row = A + i * K;
        memset(C_row, 0, K * sizeof(float));

        for (size_t p = 0; p < K; ++p) {
            float a_val = A_row[p];
            const uint32_t* B_row = B + p * K_ints;
            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t packed = B_row[g];
                float* c_out = C_row + g * 32;
                for (int b = 0; b < 32; ++b) {
                    float sign = (packed & (1u << b)) ? 1.0f : -1.0f;
                    c_out[b] += a_val * sign;
                }
            }
        }
    }
}
