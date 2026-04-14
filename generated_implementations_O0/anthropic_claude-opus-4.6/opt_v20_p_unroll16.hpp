
#pragma once
#include <cstdint>
#include <cstddef>
#include <string.h>

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, 
            float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    const int P_TILE = 16;

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

        size_t p = 0;
        for (; p + P_TILE <= K; p += P_TILE) {
            float a0[P_TILE], a1[P_TILE], a2[P_TILE], a3[P_TILE];
            const uint32_t* B_rows[P_TILE];
            for (int q = 0; q < P_TILE; ++q) {
                a0[q] = A_row0[p+q];
                a1[q] = A_row1[p+q];
                a2[q] = A_row2[p+q];
                a3[q] = A_row3[p+q];
                B_rows[q] = B + (p+q) * K_ints;
            }

            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t pk[P_TILE];
                for (int q = 0; q < P_TILE; ++q) pk[q] = B_rows[q][g];
                
                float* c0 = C_row0 + g * 32;
                float* c1 = C_row1 + g * 32;
                float* c2 = C_row2 + g * 32;
                float* c3 = C_row3 + g * 32;

                for (int b = 0; b < 32; ++b) {
                    uint32_t mask = 1u << b;
                    float v0 = 0, v1 = 0, v2 = 0, v3 = 0;
                    for (int q = 0; q < P_TILE; ++q) {
                        float s = (pk[q] & mask) ? 1.0f : -1.0f;
                        v0 += a0[q] * s;
                        v1 += a1[q] * s;
                        v2 += a2[q] * s;
                        v3 += a3[q] * s;
                    }
                    c0[b] += v0;
                    c1[b] += v1;
                    c2[b] += v2;
                    c3[b] += v3;
                }
            }
        }
        // Handle remaining p values
        for (; p < K; ++p) {
            float a0v = A_row0[p], a1v = A_row1[p], a2v = A_row2[p], a3v = A_row3[p];
            const uint32_t* B_row = B + p * K_ints;
            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t packed = B_row[g];
                float* c0 = C_row0 + g * 32;
                float* c1 = C_row1 + g * 32;
                float* c2 = C_row2 + g * 32;
                float* c3 = C_row3 + g * 32;
                for (int b = 0; b < 32; ++b) {
                    float sign = (packed & (1u << b)) ? 1.0f : -1.0f;
                    c0[b] += a0v * sign;
                    c1[b] += a1v * sign;
                    c2[b] += a2v * sign;
                    c3[b] += a3v * sign;
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
