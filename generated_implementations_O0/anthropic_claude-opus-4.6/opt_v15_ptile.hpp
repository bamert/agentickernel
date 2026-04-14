
#pragma once
#include <cstdint>
#include <cstddef>
#include <string.h>

// v7/v11 base (best so far). Try tiling along p dimension to improve 
// locality of A reads. Also try larger column tiles.
// 
// With K=3072: C row = 12KB, 4 C rows = 48KB. B row = 384 bytes.
// A row = 12KB. 4 A rows = 48KB.
// Total working set per p: 4 C rows (48KB read/write) + 4 A values + 1 B row (384B)
// The C rows dominate and 48KB fits in L1 (64KB).
// 
// The issue might be instruction throughput, not cache.
// Let me try to help the compiler by making the inner loop cleaner.

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

        for (size_t p = 0; p < K; ++p) {
            float a0 = A_row0[p];
            float a1 = A_row1[p];
            float a2 = A_row2[p];
            float a3 = A_row3[p];
            const uint32_t* B_row = B + p * K_ints;

            size_t g = 0;
            for (; g + 2 <= K_ints; g += 2) {
                uint32_t packed0 = B_row[g];
                uint32_t packed1 = B_row[g + 1];
                
                float* c0_0 = C_row0 + g * 32;
                float* c1_0 = C_row1 + g * 32;
                float* c2_0 = C_row2 + g * 32;
                float* c3_0 = C_row3 + g * 32;
                float* c0_1 = c0_0 + 32;
                float* c1_1 = c1_0 + 32;
                float* c2_1 = c2_0 + 32;
                float* c3_1 = c3_0 + 32;

                for (int b = 0; b < 32; ++b) {
                    float sign = (packed0 & (1u << b)) ? 1.0f : -1.0f;
                    c0_0[b] += a0 * sign;
                    c1_0[b] += a1 * sign;
                    c2_0[b] += a2 * sign;
                    c3_0[b] += a3 * sign;
                }
                for (int b = 0; b < 32; ++b) {
                    float sign = (packed1 & (1u << b)) ? 1.0f : -1.0f;
                    c0_1[b] += a0 * sign;
                    c1_1[b] += a1 * sign;
                    c2_1[b] += a2 * sign;
                    c3_1[b] += a3 * sign;
                }
            }
            for (; g < K_ints; ++g) {
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
