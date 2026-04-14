#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Optimization 8: Improving Opt6 by using SIMD for the accumulation
// We can process 4 bits at a time in parallel using NEON.
// Since we want to add either +val_a or -val_a for each bit,
// we can form a vector of +val_a/-val_a and add it to C.
// The bottleneck in Opt6 was likely the conditional branch.
// We can use bitwise operations to create the mask for +val_a/-val_a.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    for (size_t i = 0; i < M; ++i) {
        const float* row_A = &A[i * K];
        float* row_C = &C[i * K];

        for (size_t p = 0; p < K; ++p) {
            float val_a = row_A[p];
            float32x4_t v_a = vdupq_n_f32(val_a);
            float32x4_t v_na = vdupq_n_f32(-val_a);
            
            const uint32_t* B_row_p = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = B_row_p[j_int];
                float* c_ptr = &row_C[j_int * 32];
                
                // Process 4 bits at a time using NEON
                for (int b = 0; b < 32; b += 4) {
                    uint32_t chunk = (packed >> b) & 0xF;
                    
                    // Construct vector: map 0 to -val_a, 1 to +val_a
                    float v_vals[4];
                    v_vals[0] = (chunk & 1) ? val_a : -val_a;
                    v_vals[1] = (chunk & 2) ? val_a : -val_a;
                    v_vals[2] = (chunk & 4) ? val_a : -val_a;
                    v_vals[3] = (chunk & 8) ? val_a : -val_a;
                    
                    float32x4_t v = vld1q_f32(v_vals);
                    float32x4_t c_vec = vld1q_f32(&c_ptr[b]);
                    vst1q_f32(&c_ptr[b], vaddq_f32(c_vec, v));
                }
            }
        }
    }
}
