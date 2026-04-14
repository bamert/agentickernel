#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Optimization 3: Improved NEON approach.
// Correctly map individual bits to +1.0f / -1.0f using vector instructions.
// Process 4 bits at a time, each bit corresponds to one float.
// A block of 4 floats (for a 32-bit chunk) can be handled using vbslq_f32 or conditional additions.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M * K; ++i) {
        C[i] = 0.0f;
    }

    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t minus_one = vdupq_n_f32(-1.0f);

    for (size_t i = 0; i < M; ++i) {
        const float* row_A = &A[i * K];
        float* row_C = &C[i * K];

        for (size_t p = 0; p < K; ++p) {
            float val_a = row_A[p];
            float32x4_t v_a = vdupq_n_f32(val_a);
            
            const uint32_t* row_B_packed = &B[p * K_ints];

            for (size_t j = 0; j < K_ints; ++j) {
                uint32_t packed = row_B_packed[j];
                
                // Process 32 bits, 4 at a time (8 iterations)
                for (size_t b = 0; b < 8; ++b) {
                    uint32_t chunk = (packed >> (b * 4)) & 0xF;
                    
                    float vals[4];
                    for(int k=0; k<4; ++k) {
                        vals[k] = ((chunk >> k) & 1) ? val_a : -val_a;
                    }
                    
                    float32x4_t v_vals = vld1q_f32(vals);
                    float32x4_t* c_ptr = reinterpret_cast<float32x4_t*>(&row_C[j * 32 + b * 4]);
                    vst1q_f32((float*)c_ptr, vaddq_f32(vld1q_f32((float*)c_ptr), v_vals));
                }
            }
        }
    }
}
