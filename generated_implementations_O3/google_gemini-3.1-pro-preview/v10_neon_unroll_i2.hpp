#include <arm_neon.h>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    const uint32_t m_data[32] = {
        1u<<0, 1u<<1, 1u<<2, 1u<<3, 1u<<4, 1u<<5, 1u<<6, 1u<<7,
        1u<<8, 1u<<9, 1u<<10,1u<<11,1u<<12,1u<<13,1u<<14,1u<<15,
        1u<<16,1u<<17,1u<<18,1u<<19,1u<<20,1u<<21,1u<<22,1u<<23,
        1u<<24,1u<<25,1u<<26,1u<<27,1u<<28,1u<<29,1u<<30,1u<<31
    };

    uint32x4_t m0 = vld1q_u32(m_data + 0);
    uint32x4_t m1 = vld1q_u32(m_data + 4);
    uint32x4_t m2 = vld1q_u32(m_data + 8);
    uint32x4_t m3 = vld1q_u32(m_data + 12);
    uint32x4_t m4 = vld1q_u32(m_data + 16);
    uint32x4_t m5 = vld1q_u32(m_data + 20);
    uint32x4_t m6 = vld1q_u32(m_data + 24);
    uint32x4_t m7 = vld1q_u32(m_data + 28);

    for (size_t i = 0; i < M * K; ++i) {
        C[i] = 0.0f;
    }

    size_t PB = 512;
    for (size_t i = 0; i < M; i += 2) {
        for (size_t p_bl = 0; p_bl < K; p_bl += PB) {
            size_t p_end = p_bl + PB < K ? p_bl + PB : K;

            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                float* C_row0 = C + (i+0) * K + j_int * 32;
                float* C_row1 = C + (i+1) * K + j_int * 32;

                float32x4_t c00 = vld1q_f32(C_row0 + 0);
                float32x4_t c01 = vld1q_f32(C_row0 + 4);
                float32x4_t c02 = vld1q_f32(C_row0 + 8);
                float32x4_t c03 = vld1q_f32(C_row0 + 12);
                float32x4_t c04 = vld1q_f32(C_row0 + 16);
                float32x4_t c05 = vld1q_f32(C_row0 + 20);
                float32x4_t c06 = vld1q_f32(C_row0 + 24);
                float32x4_t c07 = vld1q_f32(C_row0 + 28);

                float32x4_t c10 = vld1q_f32(C_row1 + 0);
                float32x4_t c11 = vld1q_f32(C_row1 + 4);
                float32x4_t c12 = vld1q_f32(C_row1 + 8);
                float32x4_t c13 = vld1q_f32(C_row1 + 12);
                float32x4_t c14 = vld1q_f32(C_row1 + 16);
                float32x4_t c15 = vld1q_f32(C_row1 + 20);
                float32x4_t c16 = vld1q_f32(C_row1 + 24);
                float32x4_t c17 = vld1q_f32(C_row1 + 28);

                for (size_t p = p_bl; p < p_end; ++p) {
                    uint32_t packed = B[p * K_ints + j_int];
                    uint32x4_t vp = vdupq_n_u32(packed);

                    uint32x4_t t0 = vtstq_u32(vp, m0);
                    uint32x4_t t1 = vtstq_u32(vp, m1);
                    uint32x4_t t2 = vtstq_u32(vp, m2);
                    uint32x4_t t3 = vtstq_u32(vp, m3);
                    uint32x4_t t4 = vtstq_u32(vp, m4);
                    uint32x4_t t5 = vtstq_u32(vp, m5);
                    uint32x4_t t6 = vtstq_u32(vp, m6);
                    uint32x4_t t7 = vtstq_u32(vp, m7);

                    float32x4_t va0 = vdupq_n_f32(A[(i+0)*K + p]);
                    float32x4_t vma0 = vnegq_f32(va0);
                    c00 = vaddq_f32(c00, vbslq_f32(t0, va0, vma0));
                    c01 = vaddq_f32(c01, vbslq_f32(t1, va0, vma0));
                    c02 = vaddq_f32(c02, vbslq_f32(t2, va0, vma0));
                    c03 = vaddq_f32(c03, vbslq_f32(t3, va0, vma0));
                    c04 = vaddq_f32(c04, vbslq_f32(t4, va0, vma0));
                    c05 = vaddq_f32(c05, vbslq_f32(t5, va0, vma0));
                    c06 = vaddq_f32(c06, vbslq_f32(t6, va0, vma0));
                    c07 = vaddq_f32(c07, vbslq_f32(t7, va0, vma0));

                    float32x4_t va1 = vdupq_n_f32(A[(i+1)*K + p]);
                    float32x4_t vma1 = vnegq_f32(va1);
                    c10 = vaddq_f32(c10, vbslq_f32(t0, va1, vma1));
                    c11 = vaddq_f32(c11, vbslq_f32(t1, va1, vma1));
                    c12 = vaddq_f32(c12, vbslq_f32(t2, va1, vma1));
                    c13 = vaddq_f32(c13, vbslq_f32(t3, va1, vma1));
                    c14 = vaddq_f32(c14, vbslq_f32(t4, va1, vma1));
                    c15 = vaddq_f32(c15, vbslq_f32(t5, va1, vma1));
                    c16 = vaddq_f32(c16, vbslq_f32(t6, va1, vma1));
                    c17 = vaddq_f32(c17, vbslq_f32(t7, va1, vma1));
                }

                vst1q_f32(C_row0 + 0, c00);
                vst1q_f32(C_row0 + 4, c01);
                vst1q_f32(C_row0 + 8, c02);
                vst1q_f32(C_row0 + 12, c03);
                vst1q_f32(C_row0 + 16, c04);
                vst1q_f32(C_row0 + 20, c05);
                vst1q_f32(C_row0 + 24, c06);
                vst1q_f32(C_row0 + 28, c07);

                vst1q_f32(C_row1 + 0, c10);
                vst1q_f32(C_row1 + 4, c11);
                vst1q_f32(C_row1 + 8, c12);
                vst1q_f32(C_row1 + 12, c13);
                vst1q_f32(C_row1 + 16, c14);
                vst1q_f32(C_row1 + 20, c15);
                vst1q_f32(C_row1 + 24, c16);
                vst1q_f32(C_row1 + 28, c17);
            }
        }
    }
}
