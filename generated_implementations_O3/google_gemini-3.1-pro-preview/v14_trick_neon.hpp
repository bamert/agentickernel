#include <arm_neon.h>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    const uint32_t m_data[16] = {
        1u<<0, 1u<<1, 1u<<2, 1u<<3, 1u<<4, 1u<<5, 1u<<6, 1u<<7,
        1u<<8, 1u<<9, 1u<<10,1u<<11,1u<<12,1u<<13,1u<<14,1u<<15
    };
    const uint32_t m_data2[16] = {
        1u<<16, 1u<<17, 1u<<18, 1u<<19, 1u<<20, 1u<<21, 1u<<22, 1u<<23,
        1u<<24, 1u<<25, 1u<<26, 1u<<27, 1u<<28, 1u<<29, 1u<<30, 1u<<31
    };

    uint32x4_t m0 = vld1q_u32(m_data + 0);
    uint32x4_t m1 = vld1q_u32(m_data + 4);
    uint32x4_t m2 = vld1q_u32(m_data + 8);
    uint32x4_t m3 = vld1q_u32(m_data + 12);
    uint32x4_t m4 = vld1q_u32(m_data2 + 0);
    uint32x4_t m5 = vld1q_u32(m_data2 + 4);
    uint32x4_t m6 = vld1q_u32(m_data2 + 8);
    uint32x4_t m7 = vld1q_u32(m_data2 + 12);

    for (size_t i = 0; i < M; i += 4) {
        float S0 = 0.0f, S1 = 0.0f, S2 = 0.0f, S3 = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            S0 += A[(i+0)*K + p];
            S1 += A[(i+1)*K + p];
            S2 += A[(i+2)*K + p];
            S3 += A[(i+3)*K + p];
        }
        float32x4_t vS0 = vdupq_n_f32(S0);
        float32x4_t vS1 = vdupq_n_f32(S1);
        float32x4_t vS2 = vdupq_n_f32(S2);
        float32x4_t vS3 = vdupq_n_f32(S3);

        const float* a0_ptr = A + (i+0)*K;
        const float* a1_ptr = A + (i+1)*K;
        const float* a2_ptr = A + (i+2)*K;
        const float* a3_ptr = A + (i+3)*K;

        for (size_t j_int = 0; j_int < K_ints; ++j_int) {
            float32x4_t c00 = vdupq_n_f32(0.0f); float32x4_t c01 = c00; float32x4_t c02 = c00; float32x4_t c03 = c00;
            float32x4_t c10 = c00; float32x4_t c11 = c00; float32x4_t c12 = c00; float32x4_t c13 = c00;
            float32x4_t c20 = c00; float32x4_t c21 = c00; float32x4_t c22 = c00; float32x4_t c23 = c00;
            float32x4_t c30 = c00; float32x4_t c31 = c00; float32x4_t c32 = c00; float32x4_t c33 = c00;

            const uint32_t* b_ptr = B + j_int;
            
            #pragma unroll 4
            for (size_t p = 0; p < K; ++p) {
                uint32x4_t vp = vdupq_n_u32(*b_ptr); b_ptr += K_ints;
                
                uint32x4_t va0 = vreinterpretq_u32_f32(vdupq_n_f32(a0_ptr[p]));
                uint32x4_t va1 = vreinterpretq_u32_f32(vdupq_n_f32(a1_ptr[p]));
                uint32x4_t va2 = vreinterpretq_u32_f32(vdupq_n_f32(a2_ptr[p]));
                uint32x4_t va3 = vreinterpretq_u32_f32(vdupq_n_f32(a3_ptr[p]));

                uint32x4_t t0 = vtstq_u32(vp, m0);
                c00 = vaddq_f32(c00, vreinterpretq_f32_u32(vandq_u32(t0, va0)));
                c10 = vaddq_f32(c10, vreinterpretq_f32_u32(vandq_u32(t0, va1)));
                c20 = vaddq_f32(c20, vreinterpretq_f32_u32(vandq_u32(t0, va2)));
                c30 = vaddq_f32(c30, vreinterpretq_f32_u32(vandq_u32(t0, va3)));

                uint32x4_t t1 = vtstq_u32(vp, m1);
                c01 = vaddq_f32(c01, vreinterpretq_f32_u32(vandq_u32(t1, va0)));
                c11 = vaddq_f32(c11, vreinterpretq_f32_u32(vandq_u32(t1, va1)));
                c21 = vaddq_f32(c21, vreinterpretq_f32_u32(vandq_u32(t1, va2)));
                c31 = vaddq_f32(c31, vreinterpretq_f32_u32(vandq_u32(t1, va3)));

                uint32x4_t t2 = vtstq_u32(vp, m2);
                c02 = vaddq_f32(c02, vreinterpretq_f32_u32(vandq_u32(t2, va0)));
                c12 = vaddq_f32(c12, vreinterpretq_f32_u32(vandq_u32(t2, va1)));
                c22 = vaddq_f32(c22, vreinterpretq_f32_u32(vandq_u32(t2, va2)));
                c32 = vaddq_f32(c32, vreinterpretq_f32_u32(vandq_u32(t2, va3)));

                uint32x4_t t3 = vtstq_u32(vp, m3);
                c03 = vaddq_f32(c03, vreinterpretq_f32_u32(vandq_u32(t3, va0)));
                c13 = vaddq_f32(c13, vreinterpretq_f32_u32(vandq_u32(t3, va1)));
                c23 = vaddq_f32(c23, vreinterpretq_f32_u32(vandq_u32(t3, va2)));
                c33 = vaddq_f32(c33, vreinterpretq_f32_u32(vandq_u32(t3, va3)));
            }

            c00 = vsubq_f32(vaddq_f32(c00, c00), vS0); c01 = vsubq_f32(vaddq_f32(c01, c01), vS0);
            c02 = vsubq_f32(vaddq_f32(c02, c02), vS0); c03 = vsubq_f32(vaddq_f32(c03, c03), vS0);
            c10 = vsubq_f32(vaddq_f32(c10, c10), vS1); c11 = vsubq_f32(vaddq_f32(c11, c11), vS1);
            c12 = vsubq_f32(vaddq_f32(c12, c12), vS1); c13 = vsubq_f32(vaddq_f32(c13, c13), vS1);
            c20 = vsubq_f32(vaddq_f32(c20, c20), vS2); c21 = vsubq_f32(vaddq_f32(c21, c21), vS2);
            c22 = vsubq_f32(vaddq_f32(c22, c22), vS2); c23 = vsubq_f32(vaddq_f32(c23, c23), vS2);
            c30 = vsubq_f32(vaddq_f32(c30, c30), vS3); c31 = vsubq_f32(vaddq_f32(c31, c31), vS3);
            c32 = vsubq_f32(vaddq_f32(c32, c32), vS3); c33 = vsubq_f32(vaddq_f32(c33, c33), vS3);

            vst1q_f32(C + (i+0)*K + j_int*32 + 0, c00); vst1q_f32(C + (i+0)*K + j_int*32 + 4, c01);
            vst1q_f32(C + (i+0)*K + j_int*32 + 8, c02); vst1q_f32(C + (i+0)*K + j_int*32 + 12, c03);
            vst1q_f32(C + (i+1)*K + j_int*32 + 0, c10); vst1q_f32(C + (i+1)*K + j_int*32 + 4, c11);
            vst1q_f32(C + (i+1)*K + j_int*32 + 8, c12); vst1q_f32(C + (i+1)*K + j_int*32 + 12, c13);
            vst1q_f32(C + (i+2)*K + j_int*32 + 0, c20); vst1q_f32(C + (i+2)*K + j_int*32 + 4, c21);
            vst1q_f32(C + (i+2)*K + j_int*32 + 8, c22); vst1q_f32(C + (i+2)*K + j_int*32 + 12, c23);
            vst1q_f32(C + (i+3)*K + j_int*32 + 0, c30); vst1q_f32(C + (i+3)*K + j_int*32 + 4, c31);
            vst1q_f32(C + (i+3)*K + j_int*32 + 8, c32); vst1q_f32(C + (i+3)*K + j_int*32 + 12, c33);

            // BITS 16-31
            c00 = vdupq_n_f32(0.0f); c01 = c00; c02 = c00; c03 = c00;
            c10 = c00; c11 = c00; c12 = c00; c13 = c00;
            c20 = c00; c21 = c00; c22 = c00; c23 = c00;
            c30 = c00; c31 = c00; c32 = c00; c33 = c00;

            b_ptr = B + j_int;
            #pragma unroll 4
            for (size_t p = 0; p < K; ++p) {
                uint32x4_t vp = vdupq_n_u32(*b_ptr); b_ptr += K_ints;
                
                uint32x4_t va0 = vreinterpretq_u32_f32(vdupq_n_f32(a0_ptr[p]));
                uint32x4_t va1 = vreinterpretq_u32_f32(vdupq_n_f32(a1_ptr[p]));
                uint32x4_t va2 = vreinterpretq_u32_f32(vdupq_n_f32(a2_ptr[p]));
                uint32x4_t va3 = vreinterpretq_u32_f32(vdupq_n_f32(a3_ptr[p]));

                uint32x4_t t4 = vtstq_u32(vp, m4);
                c00 = vaddq_f32(c00, vreinterpretq_f32_u32(vandq_u32(t4, va0)));
                c10 = vaddq_f32(c10, vreinterpretq_f32_u32(vandq_u32(t4, va1)));
                c20 = vaddq_f32(c20, vreinterpretq_f32_u32(vandq_u32(t4, va2)));
                c30 = vaddq_f32(c30, vreinterpretq_f32_u32(vandq_u32(t4, va3)));

                uint32x4_t t5 = vtstq_u32(vp, m5);
                c01 = vaddq_f32(c01, vreinterpretq_f32_u32(vandq_u32(t5, va0)));
                c11 = vaddq_f32(c11, vreinterpretq_f32_u32(vandq_u32(t5, va1)));
                c21 = vaddq_f32(c21, vreinterpretq_f32_u32(vandq_u32(t5, va2)));
                c31 = vaddq_f32(c31, vreinterpretq_f32_u32(vandq_u32(t5, va3)));

                uint32x4_t t6 = vtstq_u32(vp, m6);
                c02 = vaddq_f32(c02, vreinterpretq_f32_u32(vandq_u32(t6, va0)));
                c12 = vaddq_f32(c12, vreinterpretq_f32_u32(vandq_u32(t6, va1)));
                c22 = vaddq_f32(c22, vreinterpretq_f32_u32(vandq_u32(t6, va2)));
                c32 = vaddq_f32(c32, vreinterpretq_f32_u32(vandq_u32(t6, va3)));

                uint32x4_t t7 = vtstq_u32(vp, m7);
                c03 = vaddq_f32(c03, vreinterpretq_f32_u32(vandq_u32(t7, va0)));
                c13 = vaddq_f32(c13, vreinterpretq_f32_u32(vandq_u32(t7, va1)));
                c23 = vaddq_f32(c23, vreinterpretq_f32_u32(vandq_u32(t7, va2)));
                c33 = vaddq_f32(c33, vreinterpretq_f32_u32(vandq_u32(t7, va3)));
            }

            c00 = vsubq_f32(vaddq_f32(c00, c00), vS0); c01 = vsubq_f32(vaddq_f32(c01, c01), vS0);
            c02 = vsubq_f32(vaddq_f32(c02, c02), vS0); c03 = vsubq_f32(vaddq_f32(c03, c03), vS0);
            c10 = vsubq_f32(vaddq_f32(c10, c10), vS1); c11 = vsubq_f32(vaddq_f32(c11, c11), vS1);
            c12 = vsubq_f32(vaddq_f32(c12, c12), vS1); c13 = vsubq_f32(vaddq_f32(c13, c13), vS1);
            c20 = vsubq_f32(vaddq_f32(c20, c20), vS2); c21 = vsubq_f32(vaddq_f32(c21, c21), vS2);
            c22 = vsubq_f32(vaddq_f32(c22, c22), vS2); c23 = vsubq_f32(vaddq_f32(c23, c23), vS2);
            c30 = vsubq_f32(vaddq_f32(c30, c30), vS3); c31 = vsubq_f32(vaddq_f32(c31, c31), vS3);
            c32 = vsubq_f32(vaddq_f32(c32, c32), vS3); c33 = vsubq_f32(vaddq_f32(c33, c33), vS3);

            vst1q_f32(C + (i+0)*K + j_int*32 + 16, c00); vst1q_f32(C + (i+0)*K + j_int*32 + 20, c01);
            vst1q_f32(C + (i+0)*K + j_int*32 + 24, c02); vst1q_f32(C + (i+0)*K + j_int*32 + 28, c03);
            vst1q_f32(C + (i+1)*K + j_int*32 + 16, c10); vst1q_f32(C + (i+1)*K + j_int*32 + 20, c11);
            vst1q_f32(C + (i+1)*K + j_int*32 + 24, c12); vst1q_f32(C + (i+1)*K + j_int*32 + 28, c13);
            vst1q_f32(C + (i+2)*K + j_int*32 + 16, c20); vst1q_f32(C + (i+2)*K + j_int*32 + 20, c21);
            vst1q_f32(C + (i+2)*K + j_int*32 + 24, c22); vst1q_f32(C + (i+2)*K + j_int*32 + 28, c23);
            vst1q_f32(C + (i+3)*K + j_int*32 + 16, c30); vst1q_f32(C + (i+3)*K + j_int*32 + 20, c31);
            vst1q_f32(C + (i+3)*K + j_int*32 + 24, c32); vst1q_f32(C + (i+3)*K + j_int*32 + 28, c33);
        }
    }
}
