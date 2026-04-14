#pragma once
#include <cstdint>
#include <cstddef>

// Optimized v25:
// Re-visiting the absolute fastest, v3.
// v3 had the best performance, let's look at it again.
/*
void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* rowC = &C[i * K];
        const float* rowA = &A[i * K];

        for (size_t j = 0; j < K; ++j) rowC[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            float val = rowA[p];
            float32x4_t v_val = vdupq_n_f32(val);
            float32x4_t v_neg_val = vnegq_f32(v_val);

            const uint32_t* rowB = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = rowB[j_int];
                
                for (size_t col = 0; col < 8; ++col) {
                    uint32_t bits = (packed >> (col * 4)) & 0xF;
                    
                    float32x4_t v_signs;
                    float res[4];
                    for(int k=0; k<4; ++k) {
                        res[k] = ((bits >> k) & 1) ? val : -val;
                    }
                    v_signs = vld1q_f32(res);
                    
                    float32x4_t v_c = vld1q_f32(&rowC[j_int * 32 + col * 4]);
                    vst1q_f32(&rowC[j_int * 32 + col * 4], vaddq_f32(v_c, v_signs));
                }
            }
        }
    }
}
*/
// v3 was fast because the compiler could effectively unroll the small local loops.
// Let's implement that again with minimal deviations.

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* __restrict__ rowC = &C[i * K];
        const float* __restrict__ rowA = &A[i * K];

        for (size_t j = 0; j < K; ++j) rowC[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            const float val = rowA[p];
            const float n_val = -val;
            const uint32_t* __restrict__ rowB = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                const uint32_t packed = rowB[j_int];
                
                for (size_t col = 0; col < 8; ++col) {
                    const uint32_t bits = (packed >> (col * 4)) & 0xF;
                    
                    alignas(16) float res[4];
                    for(int k=0; k<4; k++) {
                        res[k] = ((bits >> k) & 1) ? val : n_val;
                    }
                    
                    float* __restrict__ target = &rowC[j_int * 32 + col * 4];
                    float32x4_t v_c = vld1q_f32(target);
                    vst1q_f32(target, vaddq_f32(v_c, vld1q_f32(res)));
                }
            }
        }
    }
}
