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
    for (; i + 4 < M; i += 5) {
        float S0_f = 0.0f, S1_f = 0.0f, S2_f = 0.0f, S3_f = 0.0f, S4_f = 0.0f;
        float32x4_t vS0_acc = vdupq_n_f32(0.0f);
        float32x4_t vS1_acc = vdupq_n_f32(0.0f);
        float32x4_t vS2_acc = vdupq_n_f32(0.0f);
        float32x4_t vS3_acc = vdupq_n_f32(0.0f);
        float32x4_t vS4_acc = vdupq_n_f32(0.0f);
        
        size_t p = 0;
        for (; p + 3 < K; p += 4) {
            vS0_acc = vaddq_f32(vS0_acc, vld1q_f32(A + (i+0)*K + p));
            vS1_acc = vaddq_f32(vS1_acc, vld1q_f32(A + (i+1)*K + p));
            vS2_acc = vaddq_f32(vS2_acc, vld1q_f32(A + (i+2)*K + p));
            vS3_acc = vaddq_f32(vS3_acc, vld1q_f32(A + (i+3)*K + p));
            vS4_acc = vaddq_f32(vS4_acc, vld1q_f32(A + (i+4)*K + p));
        }
        for (; p < K; ++p) {
            S0_f += A[(i+0)*K + p];
            S1_f += A[(i+1)*K + p];
            S2_f += A[(i+2)*K + p];
            S3_f += A[(i+3)*K + p];
            S4_f += A[(i+4)*K + p];
        }
        float temp0[4], temp1[4], temp2[4], temp3[4], temp4[4];
        vst1q_f32(temp0, vS0_acc);
        vst1q_f32(temp1, vS1_acc);
        vst1q_f32(temp2, vS2_acc);
        vst1q_f32(temp3, vS3_acc);
        vst1q_f32(temp4, vS4_acc);
        S0_f += temp0[0] + temp0[1] + temp0[2] + temp0[3];
        S1_f += temp1[0] + temp1[1] + temp1[2] + temp1[3];
        S2_f += temp2[0] + temp2[1] + temp2[2] + temp2[3];
        S3_f += temp3[0] + temp3[1] + temp3[2] + temp3[3];
        S4_f += temp4[0] + temp4[1] + temp4[2] + temp4[3];

        float32x4_t vS0 = vdupq_n_f32(S0_f);
        float32x4_t vS1 = vdupq_n_f32(S1_f);
        float32x4_t vS2 = vdupq_n_f32(S2_f);
        float32x4_t vS3 = vdupq_n_f32(S3_f);
        float32x4_t vS4 = vdupq_n_f32(S4_f);

        const float* a0_ptr = A + (i+0)*K;
        const float* a1_ptr = A + (i+1)*K;
        const float* a2_ptr = A + (i+2)*K;
        const float* a3_ptr = A + (i+3)*K;
        const float* a4_ptr = A + (i+4)*K;

        for (size_t j_int = 0; j_int < K_ints; ++j_int) {
            float32x4_t c00 = vdupq_n_f32(0.0f); float32x4_t c01 = c00; float32x4_t c02 = c00; float32x4_t c03 = c00;
            float32x4_t c04 = c00; float32x4_t c05 = c00; float32x4_t c06 = c00; float32x4_t c07 = c00;

            float32x4_t c10 = c00; float32x4_t c11 = c00; float32x4_t c12 = c00; float32x4_t c13 = c00;
            float32x4_t c14 = c00; float32x4_t c15 = c00; float32x4_t c16 = c00; float32x4_t c17 = c00;

            float32x4_t c20 = c00; float32x4_t c21 = c00; float32x4_t c22 = c00; float32x4_t c23 = c00;
            float32x4_t c24 = c00; float32x4_t c25 = c00; float32x4_t c26 = c00; float32x4_t c27 = c00;

            float32x4_t c30 = c00; float32x4_t c31 = c00; float32x4_t c32 = c00; float32x4_t c33 = c00;
            float32x4_t c34 = c00; float32x4_t c35 = c00; float32x4_t c36 = c00; float32x4_t c37 = c00;

            float32x4_t c40 = c00; float32x4_t c41 = c00; float32x4_t c42 = c00; float32x4_t c43 = c00;
            float32x4_t c44 = c00; float32x4_t c45 = c00; float32x4_t c46 = c00; float32x4_t c47 = c00;

            const uint32_t* b_ptr = B + j_int;
            
            #pragma unroll 4
            for (p = 0; p < K; ++p) {
                uint32x4_t vp = vdupq_n_u32(*b_ptr); b_ptr += K_ints;
                
                uint32x4_t va0 = vreinterpretq_u32_f32(vdupq_n_f32(a0_ptr[p]));
                uint32x4_t va1 = vreinterpretq_u32_f32(vdupq_n_f32(a1_ptr[p]));
                uint32x4_t va2 = vreinterpretq_u32_f32(vdupq_n_f32(a2_ptr[p]));
                uint32x4_t va3 = vreinterpretq_u32_f32(vdupq_n_f32(a3_ptr[p]));
                uint32x4_t va4 = vreinterpretq_u32_f32(vdupq_n_f32(a4_ptr[p]));

                uint32x4_t t0 = vtstq_u32(vp, m0);
                c00 = vaddq_f32(c00, vreinterpretq_f32_u32(vandq_u32(t0, va0)));
                c10 = vaddq_f32(c10, vreinterpretq_f32_u32(vandq_u32(t0, va1)));
                c20 = vaddq_f32(c20, vreinterpretq_f32_u32(vandq_u32(t0, va2)));
                c30 = vaddq_f32(c30, vreinterpretq_f32_u32(vandq_u32(t0, va3)));
                c40 = vaddq_f32(c40, vreinterpretq_f32_u32(vandq_u32(t0, va4)));

                uint32x4_t t1 = vtstq_u32(vp, m1);
                c01 = vaddq_f32(c01, vreinterpretq_f32_u32(vandq_u32(t1, va0)));
                c11 = vaddq_f32(c11, vreinterpretq_f32_u32(vandq_u32(t1, va1)));
                c21 = vaddq_f32(c21, vreinterpretq_f32_u32(vandq_u32(t1, va2)));
                c31 = vaddq_f32(c31, vreinterpretq_f32_u32(vandq_u32(t1, va3)));
                c41 = vaddq_f32(c41, vreinterpretq_f32_u32(vandq_u32(t1, va4)));

                uint32x4_t t2 = vtstq_u32(vp, m2);
                c02 = vaddq_f32(c02, vreinterpretq_f32_u32(vandq_u32(t2, va0)));
                c12 = vaddq_f32(c12, vreinterpretq_f32_u32(vandq_u32(t2, va1)));
                c22 = vaddq_f32(c22, vreinterpretq_f32_u32(vandq_u32(t2, va2)));
                c32 = vaddq_f32(c32, vreinterpretq_f32_u32(vandq_u32(t2, va3)));
                c42 = vaddq_f32(c42, vreinterpretq_f32_u32(vandq_u32(t2, va4)));

                uint32x4_t t3 = vtstq_u32(vp, m3);
                c03 = vaddq_f32(c03, vreinterpretq_f32_u32(vandq_u32(t3, va0)));
                c13 = vaddq_f32(c13, vreinterpretq_f32_u32(vandq_u32(t3, va1)));
                c23 = vaddq_f32(c23, vreinterpretq_f32_u32(vandq_u32(t3, va2)));
                c33 = vaddq_f32(c33, vreinterpretq_f32_u32(vandq_u32(t3, va3)));
                c43 = vaddq_f32(c43, vreinterpretq_f32_u32(vandq_u32(t3, va4)));

                uint32x4_t t4 = vtstq_u32(vp, m4);
                c04 = vaddq_f32(c04, vreinterpretq_f32_u32(vandq_u32(t4, va0)));
                c14 = vaddq_f32(c14, vreinterpretq_f32_u32(vandq_u32(t4, va1)));
                c24 = vaddq_f32(c24, vreinterpretq_f32_u32(vandq_u32(t4, va2)));
                c34 = vaddq_f32(c34, vreinterpretq_f32_u32(vandq_u32(t4, va3)));
                c44 = vaddq_f32(c44, vreinterpretq_f32_u32(vandq_u32(t4, va4)));

                uint32x4_t t5 = vtstq_u32(vp, m5);
                c05 = vaddq_f32(c05, vreinterpretq_f32_u32(vandq_u32(t5, va0)));
                c15 = vaddq_f32(c15, vreinterpretq_f32_u32(vandq_u32(t5, va1)));
                c25 = vaddq_f32(c25, vreinterpretq_f32_u32(vandq_u32(t5, va2)));
                c35 = vaddq_f32(c35, vreinterpretq_f32_u32(vandq_u32(t5, va3)));
                c45 = vaddq_f32(c45, vreinterpretq_f32_u32(vandq_u32(t5, va4)));

                uint32x4_t t6 = vtstq_u32(vp, m6);
                c06 = vaddq_f32(c06, vreinterpretq_f32_u32(vandq_u32(t6, va0)));
                c16 = vaddq_f32(c16, vreinterpretq_f32_u32(vandq_u32(t6, va1)));
                c26 = vaddq_f32(c26, vreinterpretq_f32_u32(vandq_u32(t6, va2)));
                c36 = vaddq_f32(c36, vreinterpretq_f32_u32(vandq_u32(t6, va3)));
                c46 = vaddq_f32(c46, vreinterpretq_f32_u32(vandq_u32(t6, va4)));

                uint32x4_t t7 = vtstq_u32(vp, m7);
                c07 = vaddq_f32(c07, vreinterpretq_f32_u32(vandq_u32(t7, va0)));
                c17 = vaddq_f32(c17, vreinterpretq_f32_u32(vandq_u32(t7, va1)));
                c27 = vaddq_f32(c27, vreinterpretq_f32_u32(vandq_u32(t7, va2)));
                c37 = vaddq_f32(c37, vreinterpretq_f32_u32(vandq_u32(t7, va3)));
                c47 = vaddq_f32(c47, vreinterpretq_f32_u32(vandq_u32(t7, va4)));
            }

            c00 = vsubq_f32(vaddq_f32(c00, c00), vS0); c01 = vsubq_f32(vaddq_f32(c01, c01), vS0);
            c02 = vsubq_f32(vaddq_f32(c02, c02), vS0); c03 = vsubq_f32(vaddq_f32(c03, c03), vS0);
            c04 = vsubq_f32(vaddq_f32(c04, c04), vS0); c05 = vsubq_f32(vaddq_f32(c05, c05), vS0);
            c06 = vsubq_f32(vaddq_f32(c06, c06), vS0); c07 = vsubq_f32(vaddq_f32(c07, c07), vS0);

            c10 = vsubq_f32(vaddq_f32(c10, c10), vS1); c11 = vsubq_f32(vaddq_f32(c11, c11), vS1);
            c12 = vsubq_f32(vaddq_f32(c12, c12), vS1); c13 = vsubq_f32(vaddq_f32(c13, c13), vS1);
            c14 = vsubq_f32(vaddq_f32(c14, c14), vS1); c15 = vsubq_f32(vaddq_f32(c15, c15), vS1);
            c16 = vsubq_f32(vaddq_f32(c16, c16), vS1); c17 = vsubq_f32(vaddq_f32(c17, c17), vS1);

            c20 = vsubq_f32(vaddq_f32(c20, c20), vS2); c21 = vsubq_f32(vaddq_f32(c21, c21), vS2);
            c22 = vsubq_f32(vaddq_f32(c22, c22), vS2); c23 = vsubq_f32(vaddq_f32(c23, c23), vS2);
            c24 = vsubq_f32(vaddq_f32(c24, c24), vS2); c25 = vsubq_f32(vaddq_f32(c25, c25), vS2);
            c26 = vsubq_f32(vaddq_f32(c26, c26), vS2); c27 = vsubq_f32(vaddq_f32(c27, c27), vS2);

            c30 = vsubq_f32(vaddq_f32(c30, c30), vS3); c31 = vsubq_f32(vaddq_f32(c31, c31), vS3);
            c32 = vsubq_f32(vaddq_f32(c32, c32), vS3); c33 = vsubq_f32(vaddq_f32(c33, c33), vS3);
            c34 = vsubq_f32(vaddq_f32(c34, c34), vS3); c35 = vsubq_f32(vaddq_f32(c35, c35), vS3);
            c36 = vsubq_f32(vaddq_f32(c36, c36), vS3); c37 = vsubq_f32(vaddq_f32(c37, c37), vS3);

            c40 = vsubq_f32(vaddq_f32(c40, c40), vS4); c41 = vsubq_f32(vaddq_f32(c41, c41), vS4);
            c42 = vsubq_f32(vaddq_f32(c42, c42), vS4); c43 = vsubq_f32(vaddq_f32(c43, c43), vS4);
            c44 = vsubq_f32(vaddq_f32(c44, c44), vS4); c45 = vsubq_f32(vaddq_f32(c45, c45), vS4);
            c46 = vsubq_f32(vaddq_f32(c46, c46), vS4); c47 = vsubq_f32(vaddq_f32(c47, c47), vS4);

            vst1q_f32(C + (i+0)*K + j_int*32 + 0, c00); vst1q_f32(C + (i+0)*K + j_int*32 + 4, c01);
            vst1q_f32(C + (i+0)*K + j_int*32 + 8, c02); vst1q_f32(C + (i+0)*K + j_int*32 + 12, c03);
            vst1q_f32(C + (i+0)*K + j_int*32 + 16, c04); vst1q_f32(C + (i+0)*K + j_int*32 + 20, c05);
            vst1q_f32(C + (i+0)*K + j_int*32 + 24, c06); vst1q_f32(C + (i+0)*K + j_int*32 + 28, c07);

            vst1q_f32(C + (i+1)*K + j_int*32 + 0, c10); vst1q_f32(C + (i+1)*K + j_int*32 + 4, c11);
            vst1q_f32(C + (i+1)*K + j_int*32 + 8, c12); vst1q_f32(C + (i+1)*K + j_int*32 + 12, c13);
            vst1q_f32(C + (i+1)*K + j_int*32 + 16, c14); vst1q_f32(C + (i+1)*K + j_int*32 + 20, c15);
            vst1q_f32(C + (i+1)*K + j_int*32 + 24, c16); vst1q_f32(C + (i+1)*K + j_int*32 + 28, c17);

            vst1q_f32(C + (i+2)*K + j_int*32 + 0, c20); vst1q_f32(C + (i+2)*K + j_int*32 + 4, c21);
            vst1q_f32(C + (i+2)*K + j_int*32 + 8, c22); vst1q_f32(C + (i+2)*K + j_int*32 + 12, c23);
            vst1q_f32(C + (i+2)*K + j_int*32 + 16, c24); vst1q_f32(C + (i+2)*K + j_int*32 + 20, c25);
            vst1q_f32(C + (i+2)*K + j_int*32 + 24, c26); vst1q_f32(C + (i+2)*K + j_int*32 + 28, c27);

            vst1q_f32(C + (i+3)*K + j_int*32 + 0, c30); vst1q_f32(C + (i+3)*K + j_int*32 + 4, c31);
            vst1q_f32(C + (i+3)*K + j_int*32 + 8, c32); vst1q_f32(C + (i+3)*K + j_int*32 + 12, c33);
            vst1q_f32(C + (i+3)*K + j_int*32 + 16, c34); vst1q_f32(C + (i+3)*K + j_int*32 + 20, c35);
            vst1q_f32(C + (i+3)*K + j_int*32 + 24, c36); vst1q_f32(C + (i+3)*K + j_int*32 + 28, c37);

            vst1q_f32(C + (i+4)*K + j_int*32 + 0, c40); vst1q_f32(C + (i+4)*K + j_int*32 + 4, c41);
            vst1q_f32(C + (i+4)*K + j_int*32 + 8, c42); vst1q_f32(C + (i+4)*K + j_int*32 + 12, c43);
            vst1q_f32(C + (i+4)*K + j_int*32 + 16, c44); vst1q_f32(C + (i+4)*K + j_int*32 + 20, c45);
            vst1q_f32(C + (i+4)*K + j_int*32 + 24, c46); vst1q_f32(C + (i+4)*K + j_int*32 + 28, c47);
        }
    }
    
    for (; i < M; ++i) {
        float S0_f = 0.0f;
        for (size_t p = 0; p < K; ++p) S0_f += A[i*K + p];
        float32x4_t vS0 = vdupq_n_f32(S0_f);
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
