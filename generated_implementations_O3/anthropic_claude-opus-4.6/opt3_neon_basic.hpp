#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Zero out C
    for (size_t i = 0; i < M * K; ++i) {
        C[i] = 0.0f;
    }

    for (size_t i = 0; i < M; ++i) {
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            float32x4_t va = vdupq_n_f32(a_val);
            float32x4_t neg_va = vnegq_f32(va);
            
            float* c_row = C + i * K;
            const uint32_t* b_row = B + p * K_ints;
            
            for (size_t jb = 0; jb < K_ints; ++jb) {
                uint32_t packed = b_row[jb];
                size_t j_base = jb * 32;
                
                // Process 32 bits, 4 at a time (8 iterations of 4 floats)
                for (int chunk = 0; chunk < 8; ++chunk) {
                    // Extract 4 bits
                    float* c_ptr = c_row + j_base + chunk * 4;
                    float32x4_t c_vec = vld1q_f32(c_ptr);
                    
                    // Create mask for 4 bits
                    // bit 0
                    float32x4_t sel0 = (packed & 1) ? va : neg_va;
                    packed >>= 1;
                    float32x4_t sel1 = (packed & 1) ? va : neg_va;
                    packed >>= 1;
                    float32x4_t sel2 = (packed & 1) ? va : neg_va;
                    packed >>= 1;
                    float32x4_t sel3 = (packed & 1) ? va : neg_va;
                    packed >>= 1;
                    
                    // Combine: {sel0[0], sel1[0], sel2[0], sel3[0]}
                    float signs[4];
                    vst1q_f32(signs, sel0);
                    float s0 = signs[0];
                    vst1q_f32(signs, sel1);
                    float s1 = signs[0];
                    vst1q_f32(signs, sel2);
                    float s2 = signs[0];
                    vst1q_f32(signs, sel3);
                    float s3 = signs[0];
                    
                    float32x4_t add_vec = {s0, s1, s2, s3};
                    c_vec = vaddq_f32(c_vec, add_vec);
                    vst1q_f32(c_ptr, c_vec);
                }
            }
        }
    }
}
