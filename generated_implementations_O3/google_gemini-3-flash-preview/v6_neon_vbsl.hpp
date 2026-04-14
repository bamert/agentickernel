#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* Ci = &C[i * K];
        const float* Ai = &A[i * K];

        // Zero Initialize
        for (size_t j = 0; j < K; j += 4) {
            vst1q_f32(&Ci[j], vdupq_n_f32(0.0f));
        }

        for (size_t p = 0; p < K; ++p) {
            float a_val = Ai[p];
            float32x4_t va = vdupq_n_f32(a_val);
            float32x4_t vneg_a = vdupq_n_f32(-a_val);
            const uint32_t* Bp = &B[p * K_ints];

            for (size_t kj = 0; kj < K_ints; ++kj) {
                uint32_t bits = Bp[kj];
                float* Cij = &Ci[kj * 32];

                // Process 32 bits as 8 chunks of 4 floats
                for (int v_idx = 0; v_idx < 8; ++v_idx) {
                    uint32_t b4 = (bits >> (v_idx * 4)) & 0xF;
                    float32x4_t vc = vld1q_f32(&Cij[v_idx * 4]);
                    
                    // Bit-by-bit manipulation using NEON mask logic
                    // We can pre-calculate masks for 4 bits to avoid branching
                    uint32_t m0 = (b4 & 1) ? 0xFFFFFFFF : 0;
                    uint32_t m1 = (b4 & 2) ? 0xFFFFFFFF : 0;
                    uint32_t m2 = (b4 & 4) ? 0xFFFFFFFF : 0;
                    uint32_t m3 = (b4 & 8) ? 0xFFFFFFFF : 0;

                    uint32x4_t mask = {m0, m1, m2, m3};
                    
                    // res = mask ? va : vneg_a
                    // using bitwise selection: (mask & va) | (~mask & vneg_a)
                    float32x4_t val = vbslq_f32(mask, va, vneg_a);
                    vc = vaddq_f32(vc, val);
                    
                    vst1q_f32(&Cij[v_idx * 4], vc);
                }
            }
        }
    }
}
