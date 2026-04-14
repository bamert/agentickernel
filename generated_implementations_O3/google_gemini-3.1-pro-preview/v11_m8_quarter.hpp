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
    for (size_t i = 0; i < M; i += 8) {
        for (size_t p_bl = 0; p_bl < K; p_bl += PB) {
            size_t p_end = p_bl + PB < K ? p_bl + PB : K;

            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                float* C0 = C + (i+0)*K + j_int*32;
                float* C1 = C + (i+1)*K + j_int*32;
                float* C2 = C + (i+2)*K + j_int*32;
                float* C3 = C + (i+3)*K + j_int*32;
                float* C4 = C + (i+4)*K + j_int*32;
                float* C5 = C + (i+5)*K + j_int*32;
                float* C6 = C + (i+6)*K + j_int*32;
                float* C7 = C + (i+7)*K + j_int*32;

                const float* A0 = A + (i+0)*K;
                const float* A1 = A + (i+1)*K;
                const float* A2 = A + (i+2)*K;
                const float* A3 = A + (i+3)*K;
                const float* A4 = A + (i+4)*K;
                const float* A5 = A + (i+5)*K;
                const float* A6 = A + (i+6)*K;
                const float* A7 = A + (i+7)*K;

                // --- QUARTER 0 (bits 0-7) ---
                float32x4_t c00 = vld1q_f32(C0 + 0); float32x4_t c01 = vld1q_f32(C0 + 4);
                float32x4_t c10 = vld1q_f32(C1 + 0); float32x4_t c11 = vld1q_f32(C1 + 4);
                float32x4_t c20 = vld1q_f32(C2 + 0); float32x4_t c21 = vld1q_f32(C2 + 4);
                float32x4_t c30 = vld1q_f32(C3 + 0); float32x4_t c31 = vld1q_f32(C3 + 4);
                float32x4_t c40 = vld1q_f32(C4 + 0); float32x4_t c41 = vld1q_f32(C4 + 4);
                float32x4_t c50 = vld1q_f32(C5 + 0); float32x4_t c51 = vld1q_f32(C5 + 4);
                float32x4_t c60 = vld1q_f32(C6 + 0); float32x4_t c61 = vld1q_f32(C6 + 4);
                float32x4_t c70 = vld1q_f32(C7 + 0); float32x4_t c71 = vld1q_f32(C7 + 4);
                for (size_t p = p_bl; p < p_end; ++p) {
                    uint32x4_t vp = vdupq_n_u32(B[p * K_ints + j_int]);
                    uint32x4_t t0 = vtstq_u32(vp, m0);
                    uint32x4_t t1 = vtstq_u32(vp, m1);

                    float32x4_t va, vma;
                    va = vdupq_n_f32(A0[p]); vma = vnegq_f32(va); c00 = vaddq_f32(c00, vbslq_f32(t0, va, vma)); c01 = vaddq_f32(c01, vbslq_f32(t1, va, vma));
                    va = vdupq_n_f32(A1[p]); vma = vnegq_f32(va); c10 = vaddq_f32(c10, vbslq_f32(t0, va, vma)); c11 = vaddq_f32(c11, vbslq_f32(t1, va, vma));
                    va = vdupq_n_f32(A2[p]); vma = vnegq_f32(va); c20 = vaddq_f32(c20, vbslq_f32(t0, va, vma)); c21 = vaddq_f32(c21, vbslq_f32(t1, va, vma));
                    va = vdupq_n_f32(A3[p]); vma = vnegq_f32(va); c30 = vaddq_f32(c30, vbslq_f32(t0, va, vma)); c31 = vaddq_f32(c31, vbslq_f32(t1, va, vma));
                    va = vdupq_n_f32(A4[p]); vma = vnegq_f32(va); c40 = vaddq_f32(c40, vbslq_f32(t0, va, vma)); c41 = vaddq_f32(c41, vbslq_f32(t1, va, vma));
                    va = vdupq_n_f32(A5[p]); vma = vnegq_f32(va); c50 = vaddq_f32(c50, vbslq_f32(t0, va, vma)); c51 = vaddq_f32(c51, vbslq_f32(t1, va, vma));
                    va = vdupq_n_f32(A6[p]); vma = vnegq_f32(va); c60 = vaddq_f32(c60, vbslq_f32(t0, va, vma)); c61 = vaddq_f32(c61, vbslq_f32(t1, va, vma));
                    va = vdupq_n_f32(A7[p]); vma = vnegq_f32(va); c70 = vaddq_f32(c70, vbslq_f32(t0, va, vma)); c71 = vaddq_f32(c71, vbslq_f32(t1, va, vma));
                }
                vst1q_f32(C0 + 0, c00); vst1q_f32(C0 + 4, c01);
                vst1q_f32(C1 + 0, c10); vst1q_f32(C1 + 4, c11);
                vst1q_f32(C2 + 0, c20); vst1q_f32(C2 + 4, c21);
                vst1q_f32(C3 + 0, c30); vst1q_f32(C3 + 4, c31);
                vst1q_f32(C4 + 0, c40); vst1q_f32(C4 + 4, c41);
                vst1q_f32(C5 + 0, c50); vst1q_f32(C5 + 4, c51);
                vst1q_f32(C6 + 0, c60); vst1q_f32(C6 + 4, c61);
                vst1q_f32(C7 + 0, c70); vst1q_f32(C7 + 4, c71);

                // --- QUARTER 1 (bits 8-15) ---
                c00 = vld1q_f32(C0 + 8); c01 = vld1q_f32(C0 + 12);
                c10 = vld1q_f32(C1 + 8); c11 = vld1q_f32(C1 + 12);
                c20 = vld1q_f32(C2 + 8); c21 = vld1q_f32(C2 + 12);
                c30 = vld1q_f32(C3 + 8); c31 = vld1q_f32(C3 + 12);
                c40 = vld1q_f32(C4 + 8); c41 = vld1q_f32(C4 + 12);
                c50 = vld1q_f32(C5 + 8); c51 = vld1q_f32(C5 + 12);
                c60 = vld1q_f32(C6 + 8); c61 = vld1q_f32(C6 + 12);
                c70 = vld1q_f32(C7 + 8); c71 = vld1q_f32(C7 + 12);
                for (size_t p = p_bl; p < p_end; ++p) {
                    uint32x4_t vp = vdupq_n_u32(B[p * K_ints + j_int]);
                    uint32x4_t t2 = vtstq_u32(vp, m2);
                    uint32x4_t t3 = vtstq_u32(vp, m3);

                    float32x4_t va, vma;
                    va = vdupq_n_f32(A0[p]); vma = vnegq_f32(va); c00 = vaddq_f32(c00, vbslq_f32(t2, va, vma)); c01 = vaddq_f32(c01, vbslq_f32(t3, va, vma));
                    va = vdupq_n_f32(A1[p]); vma = vnegq_f32(va); c10 = vaddq_f32(c10, vbslq_f32(t2, va, vma)); c11 = vaddq_f32(c11, vbslq_f32(t3, va, vma));
                    va = vdupq_n_f32(A2[p]); vma = vnegq_f32(va); c20 = vaddq_f32(c20, vbslq_f32(t2, va, vma)); c21 = vaddq_f32(c21, vbslq_f32(t3, va, vma));
                    va = vdupq_n_f32(A3[p]); vma = vnegq_f32(va); c30 = vaddq_f32(c30, vbslq_f32(t2, va, vma)); c31 = vaddq_f32(c31, vbslq_f32(t3, va, vma));
                    va = vdupq_n_f32(A4[p]); vma = vnegq_f32(va); c40 = vaddq_f32(c40, vbslq_f32(t2, va, vma)); c41 = vaddq_f32(c41, vbslq_f32(t3, va, vma));
                    va = vdupq_n_f32(A5[p]); vma = vnegq_f32(va); c50 = vaddq_f32(c50, vbslq_f32(t2, va, vma)); c51 = vaddq_f32(c51, vbslq_f32(t3, va, vma));
                    va = vdupq_n_f32(A6[p]); vma = vnegq_f32(va); c60 = vaddq_f32(c60, vbslq_f32(t2, va, vma)); c61 = vaddq_f32(c61, vbslq_f32(t3, va, vma));
                    va = vdupq_n_f32(A7[p]); vma = vnegq_f32(va); c70 = vaddq_f32(c70, vbslq_f32(t2, va, vma)); c71 = vaddq_f32(c71, vbslq_f32(t3, va, vma));
                }
                vst1q_f32(C0 + 8, c00); vst1q_f32(C0 + 12, c01);
                vst1q_f32(C1 + 8, c10); vst1q_f32(C1 + 12, c11);
                vst1q_f32(C2 + 8, c20); vst1q_f32(C2 + 12, c21);
                vst1q_f32(C3 + 8, c30); vst1q_f32(C3 + 12, c31);
                vst1q_f32(C4 + 8, c40); vst1q_f32(C4 + 12, c41);
                vst1q_f32(C5 + 8, c50); vst1q_f32(C5 + 12, c51);
                vst1q_f32(C6 + 8, c60); vst1q_f32(C6 + 12, c61);
                vst1q_f32(C7 + 8, c70); vst1q_f32(C7 + 12, c71);

                // --- QUARTER 2 (bits 16-23) ---
                c00 = vld1q_f32(C0 + 16); c01 = vld1q_f32(C0 + 20);
                c10 = vld1q_f32(C1 + 16); c11 = vld1q_f32(C1 + 20);
                c20 = vld1q_f32(C2 + 16); c21 = vld1q_f32(C2 + 20);
                c30 = vld1q_f32(C3 + 16); c31 = vld1q_f32(C3 + 20);
                c40 = vld1q_f32(C4 + 16); c41 = vld1q_f32(C4 + 20);
                c50 = vld1q_f32(C5 + 16); c51 = vld1q_f32(C5 + 20);
                c60 = vld1q_f32(C6 + 16); c61 = vld1q_f32(C6 + 20);
                c70 = vld1q_f32(C7 + 16); c71 = vld1q_f32(C7 + 20);
                for (size_t p = p_bl; p < p_end; ++p) {
                    uint32x4_t vp = vdupq_n_u32(B[p * K_ints + j_int]);
                    uint32x4_t t4 = vtstq_u32(vp, m4);
                    uint32x4_t t5 = vtstq_u32(vp, m5);

                    float32x4_t va, vma;
                    va = vdupq_n_f32(A0[p]); vma = vnegq_f32(va); c00 = vaddq_f32(c00, vbslq_f32(t4, va, vma)); c01 = vaddq_f32(c01, vbslq_f32(t5, va, vma));
                    va = vdupq_n_f32(A1[p]); vma = vnegq_f32(va); c10 = vaddq_f32(c10, vbslq_f32(t4, va, vma)); c11 = vaddq_f32(c11, vbslq_f32(t5, va, vma));
                    va = vdupq_n_f32(A2[p]); vma = vnegq_f32(va); c20 = vaddq_f32(c20, vbslq_f32(t4, va, vma)); c21 = vaddq_f32(c21, vbslq_f32(t5, va, vma));
                    va = vdupq_n_f32(A3[p]); vma = vnegq_f32(va); c30 = vaddq_f32(c30, vbslq_f32(t4, va, vma)); c31 = vaddq_f32(c31, vbslq_f32(t5, va, vma));
                    va = vdupq_n_f32(A4[p]); vma = vnegq_f32(va); c40 = vaddq_f32(c40, vbslq_f32(t4, va, vma)); c41 = vaddq_f32(c41, vbslq_f32(t5, va, vma));
                    va = vdupq_n_f32(A5[p]); vma = vnegq_f32(va); c50 = vaddq_f32(c50, vbslq_f32(t4, va, vma)); c51 = vaddq_f32(c51, vbslq_f32(t5, va, vma));
                    va = vdupq_n_f32(A6[p]); vma = vnegq_f32(va); c60 = vaddq_f32(c60, vbslq_f32(t4, va, vma)); c61 = vaddq_f32(c61, vbslq_f32(t5, va, vma));
                    va = vdupq_n_f32(A7[p]); vma = vnegq_f32(va); c70 = vaddq_f32(c70, vbslq_f32(t4, va, vma)); c71 = vaddq_f32(c71, vbslq_f32(t5, va, vma));
                }
                vst1q_f32(C0 + 16, c00); vst1q_f32(C0 + 20, c01);
                vst1q_f32(C1 + 16, c10); vst1q_f32(C1 + 20, c11);
                vst1q_f32(C2 + 16, c20); vst1q_f32(C2 + 20, c21);
                vst1q_f32(C3 + 16, c30); vst1q_f32(C3 + 20, c31);
                vst1q_f32(C4 + 16, c40); vst1q_f32(C4 + 20, c41);
                vst1q_f32(C5 + 16, c50); vst1q_f32(C5 + 20, c51);
                vst1q_f32(C6 + 16, c60); vst1q_f32(C6 + 20, c61);
                vst1q_f32(C7 + 16, c70); vst1q_f32(C7 + 20, c71);

                // --- QUARTER 3 (bits 24-31) ---
                c00 = vld1q_f32(C0 + 24); c01 = vld1q_f32(C0 + 28);
                c10 = vld1q_f32(C1 + 24); c11 = vld1q_f32(C1 + 28);
                c20 = vld1q_f32(C2 + 24); c21 = vld1q_f32(C2 + 28);
                c30 = vld1q_f32(C3 + 24); c31 = vld1q_f32(C3 + 28);
                c40 = vld1q_f32(C4 + 24); c41 = vld1q_f32(C4 + 28);
                c50 = vld1q_f32(C5 + 24); c51 = vld1q_f32(C5 + 28);
                c60 = vld1q_f32(C6 + 24); c61 = vld1q_f32(C6 + 28);
                c70 = vld1q_f32(C7 + 24); c71 = vld1q_f32(C7 + 28);
                for (size_t p = p_bl; p < p_end; ++p) {
                    uint32x4_t vp = vdupq_n_u32(B[p * K_ints + j_int]);
                    uint32x4_t t6 = vtstq_u32(vp, m6);
                    uint32x4_t t7 = vtstq_u32(vp, m7);

                    float32x4_t va, vma;
                    va = vdupq_n_f32(A0[p]); vma = vnegq_f32(va); c00 = vaddq_f32(c00, vbslq_f32(t6, va, vma)); c01 = vaddq_f32(c01, vbslq_f32(t7, va, vma));
                    va = vdupq_n_f32(A1[p]); vma = vnegq_f32(va); c10 = vaddq_f32(c10, vbslq_f32(t6, va, vma)); c11 = vaddq_f32(c11, vbslq_f32(t7, va, vma));
                    va = vdupq_n_f32(A2[p]); vma = vnegq_f32(va); c20 = vaddq_f32(c20, vbslq_f32(t6, va, vma)); c21 = vaddq_f32(c21, vbslq_f32(t7, va, vma));
                    va = vdupq_n_f32(A3[p]); vma = vnegq_f32(va); c30 = vaddq_f32(c30, vbslq_f32(t6, va, vma)); c31 = vaddq_f32(c31, vbslq_f32(t7, va, vma));
                    va = vdupq_n_f32(A4[p]); vma = vnegq_f32(va); c40 = vaddq_f32(c40, vbslq_f32(t6, va, vma)); c41 = vaddq_f32(c41, vbslq_f32(t7, va, vma));
                    va = vdupq_n_f32(A5[p]); vma = vnegq_f32(va); c50 = vaddq_f32(c50, vbslq_f32(t6, va, vma)); c51 = vaddq_f32(c51, vbslq_f32(t7, va, vma));
                    va = vdupq_n_f32(A6[p]); vma = vnegq_f32(va); c60 = vaddq_f32(c60, vbslq_f32(t6, va, vma)); c61 = vaddq_f32(c61, vbslq_f32(t7, va, vma));
                    va = vdupq_n_f32(A7[p]); vma = vnegq_f32(va); c70 = vaddq_f32(c70, vbslq_f32(t6, va, vma)); c71 = vaddq_f32(c71, vbslq_f32(t7, va, vma));
                }
                vst1q_f32(C0 + 24, c00); vst1q_f32(C0 + 28, c01);
                vst1q_f32(C1 + 24, c10); vst1q_f32(C1 + 28, c11);
                vst1q_f32(C2 + 24, c20); vst1q_f32(C2 + 28, c21);
                vst1q_f32(C3 + 24, c30); vst1q_f32(C3 + 28, c31);
                vst1q_f32(C4 + 24, c40); vst1q_f32(C4 + 28, c41);
                vst1q_f32(C5 + 24, c50); vst1q_f32(C5 + 28, c51);
                vst1q_f32(C6 + 24, c60); vst1q_f32(C6 + 28, c61);
                vst1q_f32(C7 + 24, c70); vst1q_f32(C7 + 28, c71);
            }
        }
    }
}
