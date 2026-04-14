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

    const float32x4_t two = vdupq_n_f32(2.0f);
    const float32x4_t one = vdupq_n_f32(1.0f);

    for (size_t i = 0; i < M; ++i) {
        float* c_row = C + i * K;
        const float* a_row = A + i * K;
        
        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            float32x4_t va = vdupq_n_f32(a_val);
            
            const uint32_t* b_row = B + p * K_ints;
            
            for (size_t jb = 0; jb < K_ints; ++jb) {
                uint32_t packed = b_row[jb];
                size_t j_base = jb * 32;
                
                // Process 32 bits in groups of 4
                for (int chunk = 0; chunk < 8; ++chunk) {
                    float* c_ptr = c_row + j_base + chunk * 4;
                    
                    // Extract 4 consecutive bits into a uint32x4 vector
                    // bit0, bit1, bit2, bit3
                    uint32x4_t bits;
                    uint32_t b0 = (packed >> (chunk * 4)) & 0xF;
                    // Individual bits
                    uint32_t bit0 = b0 & 1;
                    uint32_t bit1 = (b0 >> 1) & 1;
                    uint32_t bit2 = (b0 >> 2) & 1;
                    uint32_t bit3 = (b0 >> 3) & 1;
                    
                    // Convert bits to float: bit ? 1.0 : -1.0 = 2*bit - 1
                    float s0 = (float)(int)(bit0 * 2) - 1.0f;
                    float s1 = (float)(int)(bit1 * 2) - 1.0f;
                    float s2 = (float)(int)(bit2 * 2) - 1.0f;
                    float s3 = (float)(int)(bit3 * 2) - 1.0f;
                    
                    float32x4_t signs = {s0, s1, s2, s3};
                    float32x4_t c_vec = vld1q_f32(c_ptr);
                    c_vec = vmlaq_f32(c_vec, va, signs);  // c += a_val * sign
                    vst1q_f32(c_ptr, c_vec);
                }
            }
        }
    }
}
