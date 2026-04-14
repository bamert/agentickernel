#include <arm_neon.h>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    const uint32_t m_data[32] = {
        1u<<0, 1u<<1, 1u<<2, 1u<<3,
        1u<<4, 1u<<5, 1u<<6, 1u<<7,
        1u<<8, 1u<<9, 1u<<10,1u<<11,
        1u<<12,1u<<13,1u<<14,1u<<15,
        1u<<16,1u<<17,1u<<18,1u<<19,
        1u<<20,1u<<21,1u<<22,1u<<23,
        1u<<24,1u<<25,1u<<26,1u<<27,
        1u<<28,1u<<29,1u<<30,1u<<31
    };

    uint32x4_t m0 = vld1q_u32(m_data + 0);
    uint32x4_t m1 = vld1q_u32(m_data + 4);
    uint32x4_t m2 = vld1q_u32(m_data + 8);
    uint32x4_t m3 = vld1q_u32(m_data + 12);
    uint32x4_t m4 = vld1q_u32(m_data + 16);
    uint32x4_t m5 = vld1q_u32(m_data + 20);
    uint32x4_t m6 = vld1q_u32(m_data + 24);
    uint32x4_t m7 = vld1q_u32(m_data + 28);

    size_t i = 0;
    for (; i + 1 < M; i += 2) {
        float S0 = 0.0f, S1 = 0.0f;
        
        float32x4_t vS0_acc = vdupq_n_f32(0.0f);
        float32x4_t vS1_acc = vdupq_n_f32(0.0f);
        size_t p = 0;
        for (; p + 3 < K; p += 4) {
            vS0_acc = vaddq_f32(vS0_acc, vld1q_f32(A + (i+0)*K + p));
            vS1_acc = vaddq_f32(vS1_acc, vld1q_f32(A + (i+1)*K + p));
        }
        for (; p < K; ++p) {
            S0 += A[(i+0)*K + p];
            S1 += A[(i+1)*K + p];
        }
        float temp0[4], temp1[4];
        vst1q_f32(temp0, vS0_acc);
        vst1q_f32(temp1, vS1_acc);
        S0 += temp0[0] + temp0[1] + temp0[2] + temp0[3];
        S1 += temp1[0] + temp1[1] + temp1[2] + temp1[3];

        float32x4_t vS0 = vdupq_n_f32(S0);
        float32x4_t vS1 = vdupq_n_f32(S1);

        const float* a0_ptr = A + (i+0)*K;
        const float* a1_ptr = A + (i+1)*K;

        for (size_t j_int = 0; j_int < K_ints; ++j_int) {
            float32x4_t c00 = vdupq_n_f32(0.0f); float32x4_t c01 = c00; float32x4_t c02 = c00; float32x4_t c03 = c00;
            float32x4_t c04 = c00; float32x4_t c05 = c00; float32x4_t c06 = c00; float32x4_t c07 = c00;

            float32x4_t c10 = c00; float32x4_t c11 = c00; float32x4_t c12 = c00; float32x4_t c13 = c00;
            float32x4_t c14 = c00; float32x4_t c15 = c00; float32x4_t c16 = c00; float32x4_t c17 = c00;

            const uint32_t* b_ptr = B + j_int;
            
            for (p = 0; p < K; ++p) {
                uint32x4_t vp = vdupq_n_u32(*b_ptr); b_ptr += K_ints;
                
                uint32x4_t va0 = vreinterpretq_u32_f32(vdupq_n_f32(a0_ptr[p]));
                uint32x4_t va1 = vreinterpretq_u32_f32(vdupq_n_f32(a1_ptr[p]));

                uint32x4_t t0 = vtstq_u32(vp, m0);
                c00 = vaddq_f32(c00, vreinterpretq_f32_u32(vandq_u32(t0, va0)));
                c10 = vaddq_f32(c10, vreinterpretq_f32_u32(vandq_u32(t0, va1)));

                uint32x4_t t1 = vtstq_u32(vp, m1);
                c01 = vaddq_f32(c01, vreinterpretq_f32_u32(vandq_u32(t1, va0)));
                c11 = vaddq_f32(c11, vreinterpretq_f32_u32(vandq_u32(t1, va1)));

                uint32x4_t t2 = vtstq_u32(vp, m2);
                c02 = vaddq_f32(c02, vreinterpretq_f32_u32(vandq_u32(t2, va0)));
                c12 = vaddq_f32(c12, vreinterpretq_f32_u32(vandq_u32(t2, va1)));

                uint32x4_t t3 = vtstq_u32(vp, m3);
                c03 = vaddq_f32(c03, vreinterpretq_f32_u32(vandq_u32(t3, va0)));
                c13 = vaddq_f32(c13, vreinterpretq_f32_u32(vandq_u32(t3, va1)));

                uint32x4_t t4 = vtstq_u32(vp, m4);
                c04 = vaddq_f32(c04, vreinterpretq_f32_u32(vandq_u32(t4, va0)));
                c14 = vaddq_f32(c14, vreinterpretq_f32_u32(vandq_u32(t4, va1)));

                uint32x4_t t5 = vtstq_u32(vp, m5);
                c05 = vaddq_f32(c05, vreinterpretq_f32_u32(vandq_u32(t5, va0)));
                c15 = vaddq_f32(c15, vreinterpretq_f32_u32(vandq_u32(t5, va1)));

                uint32x4_t t6 = vtstq_u32(vp, m6);
                c06 = vaddq_f32(c06, vreinterpretq_f32_u32(vandq_u32(t6, va0)));
                c16 = vaddq_f32(c16, vreinterpretq_f32_u32(vandq_u32(t6, va1)));

                uint32x4_t t7 = vtstq_u32(vp, m7);
                c07 = vaddq_f32(c07, vreinterpretq_f32_u32(vandq_u32(t7, va0)));
                c17 = vaddq_f32(c17, vreinterpretq_f32_u32(vandq_u32(t7, va1)));
            }

            c00 = vsubq_f32(vaddq_f32(c00, c00), vS0); c01 = vsubq_f32(vaddq_f32(c01, c01), vS0);
            c02 = vsubq_f32(vaddq_f32(c02, c02), vS0); c03 = vsubq_f32(vaddq_f32(c03, c03), vS0);
            c04 = vsubq_f32(vaddq_f32(c04, c04), vS0); c05 = vsubq_f32(vaddq_f32(c05, c05), vS0);
            c06 = vsubq_f32(vaddq_f32(c06, c06), vS0); c07 = vsubq_f32(vaddq_f32(c07, c07), vS0);

            c10 = vsubq_f32(vaddq_f32(c10, c10), vS1); c11 = vsubq_f32(vaddq_f32(c11, c11), vS1);
            c12 = vsubq_f32(vaddq_f32(c12, c12), vS1); c13 = vsubq_f32(vaddq_f32(c13, c13), vS1);
            c14 = vsubq_f32(vaddq_f32(c14, c14), vS1); c15 = vsubq_f32(vaddq_f32(c15, c15), vS1);
            c16 = vsubq_f32(vaddq_f32(c16, c16), vS1); c17 = vsubq_f32(vaddq_f32(c17, c17), vS1);

            vst1q_f32(C + (i+0)*K + j_int*32 + 0, c00); vst1q_f32(C + (i+0)*K + j_int*32 + 4, c01);
            vst1q_f32(C + (i+0)*K + j_int*32 + 8, c02); vst1q_f32(C + (i+0)*K + j_int*32 + 12, c03);
            vst1q_f32(C + (i+0)*K + j_int*32 + 16, c04); vst1q_f32(C + (i+0)*K + j_int*32 + 20, c05);
            vst1q_f32(C + (i+0)*K + j_int*32 + 24, c06); vst1q_f32(C + (i+0)*K + j_int*32 + 28, c07);

            vst1q_f32(C + (i+1)*K + j_int*32 + 0, c10); vst1q_f32(C + (i+1)*K + j_int*32 + 4, c11);
            vst1q_f32(C + (i+1)*K + j_int*32 + 8, c12); vst1q_f32(C + (i+1)*K + j_int*32 + 12, c13);
            vst1q_f32(C + (i+1)*K + j_int*32 + 16, c14); vst1q_f32(C + (i+1)*K + j_int*32 + 20, c15);
            vst1q_f32(C + (i+1)*K + j_int*32 + 24, c16); vst1q_f32(C + (i+1)*K + j_int*32 + 28, c17);
        }
    }
    
    for (; i < M; ++i) {
        float S0 = 0.0f;
        for (size_t p = 0; p < K; ++p) S0 += A[i*K + p];
        float32x4_t vS0 = vdupq_n_f32(S0);
        const float* a0_ptr = A + i*K;
        for (size_t j_int = 0; j_int < K_ints; ++j_int) {
            float32x4_t c00 = vdupq_n_f32(0.0f); float32x4_t c01 = c00; float32x4_t c02 = c00; float32x4_t c03 = c00;
            float32x4_t c04 = c00; float32x4_t c05 = c00; float32x4_t c06 = c00; float32x4_t c07 = c00;
            const uint32_t* b_ptr = B + j_int;
            for (size_t p = 0; p < K; ++p) {
                uint32x4_t vp = vdupq_n_u32(*b_ptr); b_ptr += K_ints;
                uint32x4_t va0 = vreinterpretq_u32_f32(vdupq_n_f32(a0_ptr[p]));
                c00 = vaddq_f32(c00, vreinterpretq_f32_u32(vandq_u32(vtstq_u32(vp, m0), va0)));
                c01 = vaddq_f32(c01, vreinterpretq_f32_u32(vandq_u32(vtstq_u32(vp, m1), va0)));
                c02 = vaddq_f32(c02, vreinterpretq_f32_u32(vandq_u32(vtstq_u32(vp, m2), va0)));
                c03 = vaddq_f32(c03, vreinterpretq_f32_u32(vandq_u32(vtstq_u32(vp, m3), va0)));
                c04 = vaddq_f32(c04, vreinterpretq_f32_u32(vandq_u32(vtstq_u32(vp, m4), va0)));
                c05 = vaddq_f32(c05, vreinterpretq_f32_u32(vandq_u32(vtstq_u32(vp, m5), va0)));
                c06 = vaddq_f32(c06, vreinterpretq_f32_u32(vandq_u32(vtstq_u32(vp, m6), va0)));
                c07 = vaddq_f32(c07, vreinterpretq_f32_u32(vandq_u32(vtstq_u32(vp, m7), va0)));
            }
            c00 = vsubq_f32(vaddq_f32(c00, c00), vS0); c01 = vsubq_f32(vaddq_f32(c01, c01), vS0);
            c02 = vsubq_f32(vaddq_f32(c02, c02), vS0); c03 = vsubq_f32(vaddq_f32(c03, c03), vS0);
            c04 = vsubq_f32(vaddq_f32(c04, c04), vS0); c05 = vsubq_f32(vaddq_f32(c05, c05), vS0);
            c06 = vsubq_f32(vaddq_f32(c06, c06), vS0); c07 = vsubq_f32(vaddq_f32(c07, c07), vS0);
            vst1q_f32(C + i*K + j_int*32 + 0, c00); vst1q_f32(C + i*K + j_int*32 + 4, c01);
            vst1q_f32(C + i*K + j_int*32 + 8, c02); vst1q_f32(C + i*K + j_int*32 + 12, c03);
            vst1q_f32(C + i*K + j_int*32 + 16, c04); vst1q_f32(C + i*K + j_int*32 + 20, c05);
            vst1q_f32(C + i*K + j_int*32 + 24, c06); vst1q_f32(C + i*K + j_int*32 + 28, c07);
        }
    }
}
