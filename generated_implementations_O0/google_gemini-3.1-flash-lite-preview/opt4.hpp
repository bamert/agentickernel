#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Optimization 4: Efficient bit translation using SIMD.
// Instead of creating a temporary array, use NEON bitwise operations
// to convert 32 bits into 32 floats (-1.0 or +1.0) directly.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* row_C = &C[i * K];
        const float* row_A = &A[i * K];

        // Initialize with zeros for this row
        for(size_t j = 0; j < K; ++j) row_C[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            float val_a = row_A[p];
            float32x4_t v_a = vdupq_n_f32(val_a);
            float32x4_t vn_a = vdupq_n_f32(-val_a);
            
            const uint32_t* row_B_ints = &B[p * K_ints];

            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = row_B_ints[j_int];
                
                // Process 32 bits (8 groups of 4 bits)
                for (size_t b = 0; b < 8; ++b) {
                    uint32_t chunk = (packed >> (b * 4)) & 0xF;
                    
                    float v[4];
                    for(int k=0; k<4; ++k) {
                        v[k] = (chunk & (1 << k)) ? val_a : -val_a;
                    }
                    
                    float32x4_t* c_ptr = reinterpret_cast<float32x4_t*>(&row_C[j_int * 32 + b * 4]);
                    float32x4_t c_vec = vld1q_f32(reinterpret_cast<float*>(c_ptr));
                    c_vec = vaddq_f32(c_vec, vld1q_f32(v));
                    vst1q_f32(reinterpret_cast<float*>(c_ptr), c_vec);
                }
            }
        }
    }
}
