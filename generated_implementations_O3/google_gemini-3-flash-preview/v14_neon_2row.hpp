#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    // Based on v12 results (14.21ms), 2-row tiling is excellent.
    // Let's refine the inner loop with NEON to see if we can push past 14ms.
    // We'll use a pre-calculated mask approach for the signs.
    for (size_t i = 0; i < M; i += 2) {
        float* Ci0 = &C[(i + 0) * K];
        float* Ci1 = &C[(i + 1) * K];
        const float* Ai0 = &A[(i + 0) * K];
        const float* Ai1 = &A[(i + 1) * K];

        for (size_t p = 0; p < K; ++p) {
            float32x4_t va0 = vdupq_n_f32(Ai0[p]);
            float32x4_t va1 = vdupq_n_f32(Ai1[p]);
            float32x4_t vneg_a0 = vnegq_f32(va0);
            float32x4_t vneg_a1 = vnegq_f32(va1);
            
            const uint32_t* Bp = &B[p * K_ints];
            
            for (size_t kj = 0; kj < K_ints; ++kj) {
                uint32_t bits = Bp[kj];
                float* C0 = &Ci0[kj * 32];
                float* C1 = &Ci1[kj * 32];

                for (int v = 0; v < 8; ++v) {
                    uint32_t b4 = (bits >> (v * 4)) & 0xF;
                    
                    uint32_t m[4];
                    m[0] = (b4 & 1) ? 0xFFFFFFFF : 0;
                    m[1] = (b4 & 2) ? 0xFFFFFFFF : 0;
                    m[2] = (b4 & 4) ? 0xFFFFFFFF : 0;
                    m[3] = (b4 & 8) ? 0xFFFFFFFF : 0;
                    uint32x4_t mask = vld1q_u32(m);

                    float32x4_t vc0 = vld1q_f32(C0 + v * 4);
                    float32x4_t vc1 = vld1q_f32(C1 + v * 4);

                    vc0 = vaddq_f32(vc0, vbslq_f32(mask, va0, vneg_a0));
                    vc1 = vaddq_f32(vc1, vbslq_f32(mask, va1, vneg_a1));

                    vst1q_f32(C0 + v * 4, vc0);
                    vst1q_f32(C1 + v * 4, vc1);
                }
            }
        }
    }
}
