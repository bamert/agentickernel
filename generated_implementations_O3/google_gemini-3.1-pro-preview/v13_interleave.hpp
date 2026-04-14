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

    for (size_t i = 0; i < M * K; ++i) {
        C[i] = 0.0f;
    }

    size_t PB = 256;
    for (size_t i = 0; i < M; i += 4) {
        const float* a0 = A + (i+0) * K;
        const float* a1 = A + (i+1) * K;
        const float* a2 = A + (i+2) * K;
        const float* a3 = A + (i+3) * K;

        for (size_t p_bl = 0; p_bl < K; p_bl += PB) {
            size_t p_end = p_bl + PB < K ? p_bl + PB : K;

            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                float* C_row0 = C + (i+0) * K + j_int * 32;
                float* C_row1 = C + (i+1) * K + j_int * 32;
                float* C_row2 = C + (i+2) * K + j_int * 32;
                float* C_row3 = C + (i+3) * K + j_int * 32;

                // HALF 0
                float32x4_t c00 = vld1q_f32(C_row0 + 0); float32x4_t c01 = vld1q_f32(C_row0 + 4);
                float32x4_t c02 = vld1q_f32(C_row0 + 8); float32x4_t c03 = vld1q_f32(C_row0 + 12);
                float32x4_t c10 = vld1q_f32(C_row1 + 0); float32x4_t c11 = vld1q_f32(C_row1 + 4);
                float32x4_t c12 = vld1q_f32(C_row1 + 8); float32x4_t c13 = vld1q_f32(C_row1 + 12);
                float32x4_t c20 = vld1q_f32(C_row2 + 0); float32x4_t c21 = vld1q_f32(C_row2 + 4);
                float32x4_t c22 = vld1q_f32(C_row2 + 8); float32x4_t c23 = vld1q_f32(C_row2 + 12);
                float32x4_t c30 = vld1q_f32(C_row3 + 0); float32x4_t c31 = vld1q_f32(C_row3 + 4);
                float32x4_t c32 = vld1q_f32(C_row3 + 8); float32x4_t c33 = vld1q_f32(C_row3 + 12);

                const uint32_t* b_ptr = B + p_bl * K_ints + j_int;
                
                #pragma unroll 4
                for (size_t p = p_bl; p < p_end; ++p) {
                    uint32x4_t vp = vdupq_n_u32(*b_ptr); b_ptr += K_ints;

                    float32x4_t va0 = vdupq_n_f32(a0[p]); float32x4_t vma0 = vnegq_f32(va0);
                    float32x4_t va1 = vdupq_n_f32(a1[p]); float32x4_t vma1 = vnegq_f32(va1);
                    float32x4_t va2 = vdupq_n_f32(a2[p]); float32x4_t vma2 = vnegq_f32(va2);
                    float32x4_t va3 = vdupq_n_f32(a3[p]); float32x4_t vma3 = vnegq_f32(va3);

                    uint32x4_t t0 = vtstq_u32(vp, m0);
                    c00 = vaddq_f32(c00, vbslq_f32(t0, va0, vma0));
                    c10 = vaddq_f32(c10, vbslq_f32(t0, va1, vma1));
                    c20 = vaddq_f32(c20, vbslq_f32(t0, va2, vma2));
                    c30 = vaddq_f32(c30, vbslq_f32(t0, va3, vma3));

                    uint32x4_t t1 = vtstq_u32(vp, m1);
                    c01 = vaddq_f32(c01, vbslq_f32(t1, va0, vma0));
                    c11 = vaddq_f32(c11, vbslq_f32(t1, va1, vma1));
                    c21 = vaddq_f32(c21, vbslq_f32(t1, va2, vma2));
                    c31 = vaddq_f32(c31, vbslq_f32(t1, va3, vma3));

                    uint32x4_t t2 = vtstq_u32(vp, m2);
                    c02 = vaddq_f32(c02, vbslq_f32(t2, va0, vma0));
                    c12 = vaddq_f32(c12, vbslq_f32(t2, va1, vma1));
                    c22 = vaddq_f32(c22, vbslq_f32(t2, va2, vma2));
                    c32 = vaddq_f32(c32, vbslq_f32(t2, va3, vma3));

                    uint32x4_t t3 = vtstq_u32(vp, m3);
                    c03 = vaddq_f32(c03, vbslq_f32(t3, va0, vma0));
                    c13 = vaddq_f32(c13, vbslq_f32(t3, va1, vma1));
                    c23 = vaddq_f32(c23, vbslq_f32(t3, va2, vma2));
                    c33 = vaddq_f32(c33, vbslq_f32(t3, va3, vma3));
                }

                vst1q_f32(C_row0 + 0, c00); vst1q_f32(C_row0 + 4, c01);
                vst1q_f32(C_row0 + 8, c02); vst1q_f32(C_row0 + 12, c03);
                vst1q_f32(C_row1 + 0, c10); vst1q_f32(C_row1 + 4, c11);
                vst1q_f32(C_row1 + 8, c12); vst1q_f32(C_row1 + 12, c13);
                vst1q_f32(C_row2 + 0, c20); vst1q_f32(C_row2 + 4, c21);
                vst1q_f32(C_row2 + 8, c22); vst1q_f32(C_row2 + 12, c23);
                vst1q_f32(C_row3 + 0, c30); vst1q_f32(C_row3 + 4, c31);
                vst1q_f32(C_row3 + 8, c32); vst1q_f32(C_row3 + 12, c33);

                // HALF 1
                c00 = vld1q_f32(C_row0 + 16); c01 = vld1q_f32(C_row0 + 20);
                c02 = vld1q_f32(C_row0 + 24); c03 = vld1q_f32(C_row0 + 28);
                c10 = vld1q_f32(C_row1 + 16); c11 = vld1q_f32(C_row1 + 20);
                c12 = vld1q_f32(C_row1 + 24); c13 = vld1q_f32(C_row1 + 28);
                c20 = vld1q_f32(C_row2 + 16); c21 = vld1q_f32(C_row2 + 20);
                c22 = vld1q_f32(C_row2 + 24); c23 = vld1q_f32(C_row2 + 28);
                c30 = vld1q_f32(C_row3 + 16); c31 = vld1q_f32(C_row3 + 20);
                c32 = vld1q_f32(C_row3 + 24); c33 = vld1q_f32(C_row3 + 28);

                b_ptr = B + p_bl * K_ints + j_int;
                
                #pragma unroll 4
                for (size_t p = p_bl; p < p_end; ++p) {
                    uint32x4_t vp = vdupq_n_u32(*b_ptr); b_ptr += K_ints;

                    float32x4_t va0 = vdupq_n_f32(a0[p]); float32x4_t vma0 = vnegq_f32(va0);
                    float32x4_t va1 = vdupq_n_f32(a1[p]); float32x4_t vma1 = vnegq_f32(va1);
                    float32x4_t va2 = vdupq_n_f32(a2[p]); float32x4_t vma2 = vnegq_f32(va2);
                    float32x4_t va3 = vdupq_n_f32(a3[p]); float32x4_t vma3 = vnegq_f32(va3);

                    uint32x4_t t4 = vtstq_u32(vp, m4);
                    c00 = vaddq_f32(c00, vbslq_f32(t4, va0, vma0));
                    c10 = vaddq_f32(c10, vbslq_f32(t4, va1, vma1));
                    c20 = vaddq_f32(c20, vbslq_f32(t4, va2, vma2));
                    c30 = vaddq_f32(c30, vbslq_f32(t4, va3, vma3));

                    uint32x4_t t5 = vtstq_u32(vp, m5);
                    c01 = vaddq_f32(c01, vbslq_f32(t5, va0, vma0));
                    c11 = vaddq_f32(c11, vbslq_f32(t5, va1, vma1));
                    c21 = vaddq_f32(c21, vbslq_f32(t5, va2, vma2));
                    c31 = vaddq_f32(c31, vbslq_f32(t5, va3, vma3));

                    uint32x4_t t6 = vtstq_u32(vp, m6);
                    c02 = vaddq_f32(c02, vbslq_f32(t6, va0, vma0));
                    c12 = vaddq_f32(c12, vbslq_f32(t6, va1, vma1));
                    c22 = vaddq_f32(c22, vbslq_f32(t6, va2, vma2));
                    c32 = vaddq_f32(c32, vbslq_f32(t6, va3, vma3));

                    uint32x4_t t7 = vtstq_u32(vp, m7);
                    c03 = vaddq_f32(c03, vbslq_f32(t7, va0, vma0));
                    c13 = vaddq_f32(c13, vbslq_f32(t7, va1, vma1));
                    c23 = vaddq_f32(c23, vbslq_f32(t7, va2, vma2));
                    c33 = vaddq_f32(c33, vbslq_f32(t7, va3, vma3));
                }

                vst1q_f32(C_row0 + 16, c00); vst1q_f32(C_row0 + 20, c01);
                vst1q_f32(C_row0 + 24, c02); vst1q_f32(C_row0 + 28, c03);
                vst1q_f32(C_row1 + 16, c10); vst1q_f32(C_row1 + 20, c11);
                vst1q_f32(C_row1 + 24, c12); vst1q_f32(C_row1 + 28, c13);
                vst1q_f32(C_row2 + 16, c20); vst1q_f32(C_row2 + 20, c21);
                vst1q_f32(C_row2 + 24, c22); vst1q_f32(C_row2 + 28, c23);
                vst1q_f32(C_row3 + 16, c30); vst1q_f32(C_row3 + 20, c31);
                vst1q_f32(C_row3 + 24, c32); vst1q_f32(C_row3 + 28, c33);
            }
        }
    }
}
