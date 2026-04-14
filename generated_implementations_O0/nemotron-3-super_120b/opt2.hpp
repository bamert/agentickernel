#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Calculates Matrix C = Matrix A * Matrix B (Optimized with NEON for 4-float vectors)
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        for (size_t p = 0; p < K; ++p) {     // For each column in A / row in B
            float a_val = A[i * K + p];
            const uint32_t* B_row = B + p * K_ints;
            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                uint32_t packed = B_row[chunk_idx];
                size_t base_j = chunk_idx * 32;
                // Process 4 floats at a time using NEON
                for (size_t j = 0; j < 32; j += 4) {
                    // Load 4 floats from C
                    float32x4_t c_vec = vld1q_f32(C + i * K + base_j + j);
                    // Extract 4 bits from packed at positions j..j+3
                    uint32_t bits4 = (packed >> j) & 0xF;
                    // Create a vector of signs: each bit -> 1.0f if set, -1.0f if not
                    // We'll build the vector lane by lane
                    float32x4_t sign_vec = vdupq_n_f32(-1.0f); // Start with -1.0f in all lanes
                    // For each bit that is set, add 2.0f to the corresponding lane
                    if (bits4 & 0x1) sign_vec = vsetq_lane_f32(1.0f, sign_vec, 0);
                    if (bits4 & 0x2) sign_vec = vsetq_lane_f32(1.0f, sign_vec, 1);
                    if (bits4 & 0x4) sign_vec = vsetq_lane_f32(1.0f, sign_vec, 2);
                    if (bits4 & 0x8) sign_vec = vsetq_lane_f32(1.0f, sign_vec, 3);
                    // Now: c_vec += a_val * sign_vec
                    c_vec = vaddq_f32(c_vec, vmulq_n_f32(sign_vec, a_val));
                    // Store back
                    vst1q_f32(C + i * K + base_j + j, c_vec);
                }
            }
        }
    }
}