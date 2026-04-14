
#pragma once
#include <cstdint>
#include <cstddef>
#include <string.h>

// Try M=2 tile with p=8 unroll to see if less register pressure helps
void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, 
            float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    size_t i = 0;
    for (; i + 2 <= M; i += 2) {
        float* __restrict__ C_row0 = C + (i + 0) * K;
        float* __restrict__ C_row1 = C + (i + 1) * K;
        const float* __restrict__ A_row0 = A + (i + 0) * K;
        const float* __restrict__ A_row1 = A + (i + 1) * K;

        memset(C_row0, 0, K * sizeof(float));
        memset(C_row1, 0, K * sizeof(float));

        size_t p = 0;
        for (; p + 8 <= K; p += 8) {
            float a00 = A_row0[p+0], a01 = A_row0[p+1], a02 = A_row0[p+2], a03 = A_row0[p+3];
            float a04 = A_row0[p+4], a05 = A_row0[p+5], a06 = A_row0[p+6], a07 = A_row0[p+7];
            float a10 = A_row1[p+0], a11 = A_row1[p+1], a12 = A_row1[p+2], a13 = A_row1[p+3];
            float a14 = A_row1[p+4], a15 = A_row1[p+5], a16 = A_row1[p+6], a17 = A_row1[p+7];
            
            const uint32_t* Bp0 = B + (p+0)*K_ints;
            const uint32_t* Bp1 = B + (p+1)*K_ints;
            const uint32_t* Bp2 = B + (p+2)*K_ints;
            const uint32_t* Bp3 = B + (p+3)*K_ints;
            const uint32_t* Bp4 = B + (p+4)*K_ints;
            const uint32_t* Bp5 = B + (p+5)*K_ints;
            const uint32_t* Bp6 = B + (p+6)*K_ints;
            const uint32_t* Bp7 = B + (p+7)*K_ints;

            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t pk0 = Bp0[g], pk1 = Bp1[g], pk2 = Bp2[g], pk3 = Bp3[g];
                uint32_t pk4 = Bp4[g], pk5 = Bp5[g], pk6 = Bp6[g], pk7 = Bp7[g];
                
                float* c0 = C_row0 + g * 32;
                float* c1 = C_row1 + g * 32;

                for (int b = 0; b < 32; ++b) {
                    uint32_t mask = 1u << b;
                    float s0 = (pk0 & mask) ? 1.0f : -1.0f;
                    float s1 = (pk1 & mask) ? 1.0f : -1.0f;
                    float s2 = (pk2 & mask) ? 1.0f : -1.0f;
                    float s3 = (pk3 & mask) ? 1.0f : -1.0f;
                    float s4 = (pk4 & mask) ? 1.0f : -1.0f;
                    float s5 = (pk5 & mask) ? 1.0f : -1.0f;
                    float s6 = (pk6 & mask) ? 1.0f : -1.0f;
                    float s7 = (pk7 & mask) ? 1.0f : -1.0f;
                    
                    c0[b] += a00*s0 + a01*s1 + a02*s2 + a03*s3 
                           + a04*s4 + a05*s5 + a06*s6 + a07*s7;
                    c1[b] += a10*s0 + a11*s1 + a12*s2 + a13*s3 
                           + a14*s4 + a15*s5 + a16*s6 + a17*s7;
                }
            }
        }
        for (; p < K; ++p) {
            float a0v = A_row0[p], a1v = A_row1[p];
            const uint32_t* B_row = B + p * K_ints;
            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t packed = B_row[g];
                float* c0 = C_row0 + g * 32;
                float* c1 = C_row1 + g * 32;
                for (int b = 0; b < 32; ++b) {
                    float sign = (packed & (1u << b)) ? 1.0f : -1.0f;
                    c0[b] += a0v * sign;
                    c1[b] += a1v * sign;
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
