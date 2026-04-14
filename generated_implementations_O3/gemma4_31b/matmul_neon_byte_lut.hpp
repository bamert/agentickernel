#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Optimized Matrix C = Matrix A * Matrix B
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    const uint8_t* B_bytes = reinterpret_cast<const uint8_t*>(B);

    struct SignEntry {
        float32x4_t v0;
        float32x4_t v1;
    };
    SignEntry sign_lut[256];

    for (int i = 0; i < 256; ++i) {
        float s0[4], s1[4];
        for (int b = 0; b < 4; ++b) {
            s0[b] = ((i >> b) & 1) ? 1.0f : -1.0f;
            s1[b] = ((i >> (b + 4)) & 1) ? 1.0f : -1.0f;
        }
        sign_lut[i].v0 = vld1q_f32(s0);
        sign_lut[i].v1 = vld1q_f32(s1);
    }

    for (size_t i = 0; i < M; i += 2) {
        const float* rowA0 = &A[i * K];
        const float* rowA1 = (i + 1 < M) ? &A[(i + 1) * K] : nullptr;
        
        for (size_t j_block = 0; j_block < K_ints; ++j_block) {
            float32x4_t acc0[8], acc1[8];
            for(int n=0; n<8; ++n) {
                acc0[n] = vdupq_n_f32(0.0f);
                acc1[n] = vdupq_n_f32(0.0f);
            }

            const size_t b_offset = j_block * 4; // shifted to byte offset for the block
            // But B is row-major: B[p][j_block], so we need a different stride.
            // B_bytes[ (p * K_ints + j_block) * 4 + offset ]
            
            for (size_t p = 0; p < K; p += 4) {
                const float32x4_t va0_0 = vdupq_n_f32(rowA0[p]);
                const float32x4_t va0_1 = vdupq_n_f32(rowA0[p+1]);
                const float32x4_t va0_2 = vdupq_n_f32(rowA0[p+2]);
                const float32x4_t va0_3 = vdupq_n_f32(rowA0[p+3]);
                
                float32x4_t va1_0 = vdupq_n_f32(0.0f), va1_1 = vdupq_n_f32(0.0f), va1_2 = vdupq_n_f32(0.0f), va1_3 = vdupq_n_f32(0.0f);
                if (rowA1) {
                    va1_0 = vdupq_n_f32(rowA1[p]);
                    va1_1 = vdupq_n_f32(rowA1[p+1]);
                    va1_2 = vdupq_n_f32(rowA1[p+2]);
                    va1_3 = vdupq_n_f32(rowA1[p+3]);
                }

                // For each p in the unroll, we access the bytes of the packed uint32
                for(int n=0; n<4; ++n) {
                    const uint8_t* b_ptr = &B_bytes[(p + n) * K_ints * 4 + j_block * 4];
                    const uint8_t b0 = b_ptr[0];
                    const uint8_t b1 = b_ptr[1];
                    const uint8_t b2 = b_ptr[2];
                    const uint8_t b3 = b_ptr[3];
                    
                    const float32x4_t va0 = (n==0)?va0_0:(n==1)?va0_1:(n==2)?va0_2:va0_3;
                    acc0[0] = vmlaq_f32(acc0[0], sign_lut[b0].v0, va0);
                    acc0[1] = vmlaq_f32(acc0[1], sign_lut[b0].v1, va0);
                    acc0[2] = vmlaq_f32(acc0[2], sign_lut[b1].v0, va0);
                    acc0[3] = vmlaq_f32(acc0[3], sign_lut[b1].v1, va0);
                    acc0[4] = vmlaq_f32(acc0[4], sign_lut[b2].v0, va0);
                    acc0[5] = vmlaq_f32(acc0[5], sign_lut[b2].v1, va0);
                    acc0[6] = vmlaq_f32(acc0[6], sign_lut[b3].v0, va0);
                    acc0[7] = vmlaq_f32(acc0[7], sign_lut[b3].v1, va0);

                    if (rowA1) {
                        const float32x4_t va1 = (n==0)?va1_0:(n==1)?va1_1:(n==2)?va1_2:va1_3;
                        acc1[0] = vmlaq_f32(acc1[0], sign_lut[b0].v0, va1);
                        acc1[1] = vmlaq_f32(acc1[1], sign_lut[b0].v1, va1);
                        acc1[2] = vmlaq_f32(acc1[2], sign_lut[b1].v0, va1);
                        acc1[3] = vmlaq_f32(acc1[3], sign_lut[b1].v1, va1);
                        acc1[4] = vmlaq_f32(acc1[4], sign_lut[b2].v0, va1);
                        acc1[5] = vmlaq_f32(acc1[5], sign_lut[b2].v1, va1);
                        acc1[6] = vmlaq_f32(acc1[6], sign_lut[b3].v0, va1);
                        acc1[7] = vmlaq_f32(acc1[7], sign_lut[b3].v1, va1);
                    }
                }
            }

            float* out_ptr0 = &C[i * K + j_block * 32];
            for(int n=0; n<8; ++n) vst1q_f32(out_ptr0 + n*4, acc0[n]);
            if (i + 1 < M) {
                float* out_ptr1 = &C[(i + 1) * K + j_block * 32];
                for(int n=0; n<8; ++n) vst1q_f32(out_ptr1 + n*4, acc1[n]);
            }
        }
    }
}
