#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Zero out C
    for (size_t idx = 0; idx < M * K; idx += 4) {
        vst1q_f32(C + idx, vdupq_n_f32(0.0f));
    }

    for (size_t i = 0; i < M; ++i) {
        float* c_row = C + i * K;
        const float* a_row = A + i * K;
        
        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            float32x4_t va = vdupq_n_f32(a_val);
            float32x4_t neg_va = vnegq_f32(va);
            
            const uint32_t* b_row = B + p * K_ints;
            
            for (size_t jb = 0; jb < K_ints; ++jb) {
                uint32_t packed = b_row[jb];
                float* c_ptr = c_row + jb * 32;
                
                // Process 32 bits using NEON bit select
                // For each group of 4 bits, create a mask and use vbslq
                for (int chunk = 0; chunk < 8; ++chunk) {
                    float32x4_t c_vec = vld1q_f32(c_ptr);
                    
                    // Extract 4 bits and expand to 32-bit masks
                    uint32_t b4 = (packed >> (chunk * 4)) & 0xF;
                    
                    // Create mask: if bit set, all 1s; else all 0s
                    uint32x4_t mask;
                    uint32_t m0 = (b4 & 1) ? 0xFFFFFFFF : 0;
                    uint32_t m1 = (b4 & 2) ? 0xFFFFFFFF : 0;
                    uint32_t m2 = (b4 & 4) ? 0xFFFFFFFF : 0;
                    uint32_t m3 = (b4 & 8) ? 0xFFFFFFFF : 0;
                    
                    uint32_t marr[4] = {m0, m1, m2, m3};
                    mask = vld1q_u32(marr);
                    
                    // Select: mask ? va : neg_va
                    float32x4_t selected = vbslq_f32(mask, va, neg_va);
                    c_vec = vaddq_f32(c_vec, selected);
                    vst1q_f32(c_ptr, c_vec);
                    
                    c_ptr += 4;
                }
            }
        }
    }
}
