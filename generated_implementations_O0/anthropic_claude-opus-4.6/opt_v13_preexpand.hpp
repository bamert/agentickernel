
#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>
#include <string.h>

// Best approach so far: v7/v11 style with 4-row tiling.
// New idea: pre-expand B into sign arrays to avoid per-bit extraction in the hot loop.
// B is K rows × K_ints words. Total bits = K×K. 
// Expanded: K×K floats of +1/-1. For K=3072: 3072*3072*4 = 36MB. Too big.
//
// Alternative: expand one B row at a time (K floats = 12KB for K=3072, fits in L1).
// For each p: expand B[p] to float signs[K], then do vectorized FMA on C rows.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Temporary buffer to hold expanded signs for one B row
    float signs[4096]; // Max K we'd see; adjust if needed. K=3072 fits.
    
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

        // Zero C rows
        memset(C_row0, 0, K * sizeof(float));
        memset(C_row1, 0, K * sizeof(float));
        memset(C_row2, 0, K * sizeof(float));
        memset(C_row3, 0, K * sizeof(float));

        for (size_t p = 0; p < K; ++p) {
            // Expand B row p to float signs
            const uint32_t* B_row = B + p * K_ints;
            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t packed = B_row[g];
                float* s = signs + g * 32;
                for (int b = 0; b < 32; ++b) {
                    s[b] = (packed & (1u << b)) ? 1.0f : -1.0f;
                }
            }

            float a0 = A_row0[p];
            float a1 = A_row1[p];
            float a2 = A_row2[p];
            float a3 = A_row3[p];
            
            float32x4_t va0 = vdupq_n_f32(a0);
            float32x4_t va1 = vdupq_n_f32(a1);
            float32x4_t va2 = vdupq_n_f32(a2);
            float32x4_t va3 = vdupq_n_f32(a3);

            // Now do vectorized FMA: C_row += a * signs
            for (size_t j = 0; j < K; j += 16) {
                float32x4_t s0 = vld1q_f32(signs + j);
                float32x4_t s1 = vld1q_f32(signs + j + 4);
                float32x4_t s2 = vld1q_f32(signs + j + 8);
                float32x4_t s3 = vld1q_f32(signs + j + 12);
                
                vst1q_f32(C_row0 + j,      vfmaq_f32(vld1q_f32(C_row0 + j),      va0, s0));
                vst1q_f32(C_row0 + j + 4,  vfmaq_f32(vld1q_f32(C_row0 + j + 4),  va0, s1));
                vst1q_f32(C_row0 + j + 8,  vfmaq_f32(vld1q_f32(C_row0 + j + 8),  va0, s2));
                vst1q_f32(C_row0 + j + 12, vfmaq_f32(vld1q_f32(C_row0 + j + 12), va0, s3));
                
                vst1q_f32(C_row1 + j,      vfmaq_f32(vld1q_f32(C_row1 + j),      va1, s0));
                vst1q_f32(C_row1 + j + 4,  vfmaq_f32(vld1q_f32(C_row1 + j + 4),  va1, s1));
                vst1q_f32(C_row1 + j + 8,  vfmaq_f32(vld1q_f32(C_row1 + j + 8),  va1, s2));
                vst1q_f32(C_row1 + j + 12, vfmaq_f32(vld1q_f32(C_row1 + j + 12), va1, s3));
                
                vst1q_f32(C_row2 + j,      vfmaq_f32(vld1q_f32(C_row2 + j),      va2, s0));
                vst1q_f32(C_row2 + j + 4,  vfmaq_f32(vld1q_f32(C_row2 + j + 4),  va2, s1));
                vst1q_f32(C_row2 + j + 8,  vfmaq_f32(vld1q_f32(C_row2 + j + 8),  va2, s2));
                vst1q_f32(C_row2 + j + 12, vfmaq_f32(vld1q_f32(C_row2 + j + 12), va2, s3));
                
                vst1q_f32(C_row3 + j,      vfmaq_f32(vld1q_f32(C_row3 + j),      va3, s0));
                vst1q_f32(C_row3 + j + 4,  vfmaq_f32(vld1q_f32(C_row3 + j + 4),  va3, s1));
                vst1q_f32(C_row3 + j + 8,  vfmaq_f32(vld1q_f32(C_row3 + j + 8),  va3, s2));
                vst1q_f32(C_row3 + j + 12, vfmaq_f32(vld1q_f32(C_row3 + j + 12), va3, s3));
            }
        }
    }

    for (; i < M; ++i) {
        float* C_row = C + i * K;
        const float* A_row = A + i * K;
        memset(C_row, 0, K * sizeof(float));

        for (size_t p = 0; p < K; ++p) {
            const uint32_t* B_row = B + p * K_ints;
            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t packed = B_row[g];
                float* s = signs + g * 32;
                for (int b = 0; b < 32; ++b) {
                    s[b] = (packed & (1u << b)) ? 1.0f : -1.0f;
                }
            }

            float a_val = A_row[p];
            float32x4_t va = vdupq_n_f32(a_val);
            
            for (size_t j = 0; j < K; j += 16) {
                float32x4_t s0 = vld1q_f32(signs + j);
                float32x4_t s1 = vld1q_f32(signs + j + 4);
                float32x4_t s2 = vld1q_f32(signs + j + 8);
                float32x4_t s3 = vld1q_f32(signs + j + 12);
                
                vst1q_f32(C_row + j,      vfmaq_f32(vld1q_f32(C_row + j),      va, s0));
                vst1q_f32(C_row + j + 4,  vfmaq_f32(vld1q_f32(C_row + j + 4),  va, s1));
                vst1q_f32(C_row + j + 8,  vfmaq_f32(vld1q_f32(C_row + j + 8),  va, s2));
                vst1q_f32(C_row + j + 12, vfmaq_f32(vld1q_f32(C_row + j + 12), va, s3));
            }
        }
    }
}
