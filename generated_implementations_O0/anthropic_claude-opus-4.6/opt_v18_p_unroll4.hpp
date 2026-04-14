
#pragma once
#include <cstdint>
#include <cstddef>
#include <string.h>

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

        // Process 4 p values at a time
        size_t p = 0;
        for (; p + 4 <= K; p += 4) {
            float a00 = A_row0[p], a01 = A_row0[p+1], a02 = A_row0[p+2], a03 = A_row0[p+3];
            float a10 = A_row1[p], a11 = A_row1[p+1], a12 = A_row1[p+2], a13 = A_row1[p+3];
            float a20 = A_row2[p], a21 = A_row2[p+1], a22 = A_row2[p+2], a23 = A_row2[p+3];
            float a30 = A_row3[p], a31 = A_row3[p+1], a32 = A_row3[p+2], a33 = A_row3[p+3];
            const uint32_t* B_row0 = B + (p+0) * K_ints;
            const uint32_t* B_row1 = B + (p+1) * K_ints;
            const uint32_t* B_row2 = B + (p+2) * K_ints;
            const uint32_t* B_row3 = B + (p+3) * K_ints;

            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t pk0 = B_row0[g];
                uint32_t pk1 = B_row1[g];
                uint32_t pk2 = B_row2[g];
                uint32_t pk3 = B_row3[g];
                float* c0 = C_row0 + g * 32;
                float* c1 = C_row1 + g * 32;
                float* c2 = C_row2 + g * 32;
                float* c3 = C_row3 + g * 32;

                for (int b = 0; b < 32; ++b) {
                    uint32_t mask = 1u << b;
                    float s0 = (pk0 & mask) ? 1.0f : -1.0f;
                    float s1 = (pk1 & mask) ? 1.0f : -1.0f;
                    float s2 = (pk2 & mask) ? 1.0f : -1.0f;
                    float s3 = (pk3 & mask) ? 1.0f : -1.0f;
                    
                    float v0 = a00*s0 + a01*s1 + a02*s2 + a03*s3;
                    float v1 = a10*s0 + a11*s1 + a12*s2 + a13*s3;
                    float v2 = a20*s0 + a21*s1 + a22*s2 + a23*s3;
                    float v3 = a30*s0 + a31*s1 + a32*s2 + a33*s3;
                    
                    c0[b] += v0;
                    c1[b] += v1;
                    c2[b] += v2;
                    c3[b] += v3;
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
