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

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }
        for (size_t p = 0; p + 7 < K; p += 8) {
            float32x4_t va0 = vdupq_n_f32(A[i * K + p + 0]);
            float32x4_t va1 = vdupq_n_f32(A[i * K + p + 1]);
            float32x4_t va2 = vdupq_n_f32(A[i * K + p + 2]);
            float32x4_t va3 = vdupq_n_f32(A[i * K + p + 3]);
            float32x4_t va4 = vdupq_n_f32(A[i * K + p + 4]);
            float32x4_t va5 = vdupq_n_f32(A[i * K + p + 5]);
            float32x4_t va6 = vdupq_n_f32(A[i * K + p + 6]);
            float32x4_t va7 = vdupq_n_f32(A[i * K + p + 7]);

            float32x4_t vma0 = vnegq_f32(va0);
            float32x4_t vma1 = vnegq_f32(va1);
            float32x4_t vma2 = vnegq_f32(va2);
            float32x4_t vma3 = vnegq_f32(va3);
            float32x4_t vma4 = vnegq_f32(va4);
            float32x4_t vma5 = vnegq_f32(va5);
            float32x4_t vma6 = vnegq_f32(va6);
            float32x4_t vma7 = vnegq_f32(va7);

            const uint32_t* B0 = B + (p + 0) * K_ints;
            const uint32_t* B1 = B + (p + 1) * K_ints;
            const uint32_t* B2 = B + (p + 2) * K_ints;
            const uint32_t* B3 = B + (p + 3) * K_ints;
            const uint32_t* B4 = B + (p + 4) * K_ints;
            const uint32_t* B5 = B + (p + 5) * K_ints;
            const uint32_t* B6 = B + (p + 6) * K_ints;
            const uint32_t* B7 = B + (p + 7) * K_ints;
            
            float* C_row = C + i * K;

            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                float32x4_t c0 = vld1q_f32(C_row + j_int * 32 + 0);
                float32x4_t c1 = vld1q_f32(C_row + j_int * 32 + 4);
                float32x4_t c2 = vld1q_f32(C_row + j_int * 32 + 8);
                float32x4_t c3 = vld1q_f32(C_row + j_int * 32 + 12);
                float32x4_t c4 = vld1q_f32(C_row + j_int * 32 + 16);
                float32x4_t c5 = vld1q_f32(C_row + j_int * 32 + 20);
                float32x4_t c6 = vld1q_f32(C_row + j_int * 32 + 24);
                float32x4_t c7 = vld1q_f32(C_row + j_int * 32 + 28);

                uint32x4_t vp0 = vdupq_n_u32(B0[j_int]);
                c0 = vaddq_f32(c0, vbslq_f32(vtstq_u32(vp0, m0), va0, vma0));
                c1 = vaddq_f32(c1, vbslq_f32(vtstq_u32(vp0, m1), va0, vma0));
                c2 = vaddq_f32(c2, vbslq_f32(vtstq_u32(vp0, m2), va0, vma0));
                c3 = vaddq_f32(c3, vbslq_f32(vtstq_u32(vp0, m3), va0, vma0));
                c4 = vaddq_f32(c4, vbslq_f32(vtstq_u32(vp0, m4), va0, vma0));
                c5 = vaddq_f32(c5, vbslq_f32(vtstq_u32(vp0, m5), va0, vma0));
                c6 = vaddq_f32(c6, vbslq_f32(vtstq_u32(vp0, m6), va0, vma0));
                c7 = vaddq_f32(c7, vbslq_f32(vtstq_u32(vp0, m7), va0, vma0));

                uint32x4_t vp1 = vdupq_n_u32(B1[j_int]);
                c0 = vaddq_f32(c0, vbslq_f32(vtstq_u32(vp1, m0), va1, vma1));
                c1 = vaddq_f32(c1, vbslq_f32(vtstq_u32(vp1, m1), va1, vma1));
                c2 = vaddq_f32(c2, vbslq_f32(vtstq_u32(vp1, m2), va1, vma1));
                c3 = vaddq_f32(c3, vbslq_f32(vtstq_u32(vp1, m3), va1, vma1));
                c4 = vaddq_f32(c4, vbslq_f32(vtstq_u32(vp1, m4), va1, vma1));
                c5 = vaddq_f32(c5, vbslq_f32(vtstq_u32(vp1, m5), va1, vma1));
                c6 = vaddq_f32(c6, vbslq_f32(vtstq_u32(vp1, m6), va1, vma1));
                c7 = vaddq_f32(c7, vbslq_f32(vtstq_u32(vp1, m7), va1, vma1));

                uint32x4_t vp2 = vdupq_n_u32(B2[j_int]);
                c0 = vaddq_f32(c0, vbslq_f32(vtstq_u32(vp2, m0), va2, vma2));
                c1 = vaddq_f32(c1, vbslq_f32(vtstq_u32(vp2, m1), va2, vma2));
                c2 = vaddq_f32(c2, vbslq_f32(vtstq_u32(vp2, m2), va2, vma2));
                c3 = vaddq_f32(c3, vbslq_f32(vtstq_u32(vp2, m3), va2, vma2));
                c4 = vaddq_f32(c4, vbslq_f32(vtstq_u32(vp2, m4), va2, vma2));
                c5 = vaddq_f32(c5, vbslq_f32(vtstq_u32(vp2, m5), va2, vma2));
                c6 = vaddq_f32(c6, vbslq_f32(vtstq_u32(vp2, m6), va2, vma2));
                c7 = vaddq_f32(c7, vbslq_f32(vtstq_u32(vp2, m7), va2, vma2));

                uint32x4_t vp3 = vdupq_n_u32(B3[j_int]);
                c0 = vaddq_f32(c0, vbslq_f32(vtstq_u32(vp3, m0), va3, vma3));
                c1 = vaddq_f32(c1, vbslq_f32(vtstq_u32(vp3, m1), va3, vma3));
                c2 = vaddq_f32(c2, vbslq_f32(vtstq_u32(vp3, m2), va3, vma3));
                c3 = vaddq_f32(c3, vbslq_f32(vtstq_u32(vp3, m3), va3, vma3));
                c4 = vaddq_f32(c4, vbslq_f32(vtstq_u32(vp3, m4), va3, vma3));
                c5 = vaddq_f32(c5, vbslq_f32(vtstq_u32(vp3, m5), va3, vma3));
                c6 = vaddq_f32(c6, vbslq_f32(vtstq_u32(vp3, m6), va3, vma3));
                c7 = vaddq_f32(c7, vbslq_f32(vtstq_u32(vp3, m7), va3, vma3));

                uint32x4_t vp4 = vdupq_n_u32(B4[j_int]);
                c0 = vaddq_f32(c0, vbslq_f32(vtstq_u32(vp4, m0), va4, vma4));
                c1 = vaddq_f32(c1, vbslq_f32(vtstq_u32(vp4, m1), va4, vma4));
                c2 = vaddq_f32(c2, vbslq_f32(vtstq_u32(vp4, m2), va4, vma4));
                c3 = vaddq_f32(c3, vbslq_f32(vtstq_u32(vp4, m3), va4, vma4));
                c4 = vaddq_f32(c4, vbslq_f32(vtstq_u32(vp4, m4), va4, vma4));
                c5 = vaddq_f32(c5, vbslq_f32(vtstq_u32(vp4, m5), va4, vma4));
                c6 = vaddq_f32(c6, vbslq_f32(vtstq_u32(vp4, m6), va4, vma4));
                c7 = vaddq_f32(c7, vbslq_f32(vtstq_u32(vp4, m7), va4, vma4));

                uint32x4_t vp5 = vdupq_n_u32(B5[j_int]);
                c0 = vaddq_f32(c0, vbslq_f32(vtstq_u32(vp5, m0), va5, vma5));
                c1 = vaddq_f32(c1, vbslq_f32(vtstq_u32(vp5, m1), va5, vma5));
                c2 = vaddq_f32(c2, vbslq_f32(vtstq_u32(vp5, m2), va5, vma5));
                c3 = vaddq_f32(c3, vbslq_f32(vtstq_u32(vp5, m3), va5, vma5));
                c4 = vaddq_f32(c4, vbslq_f32(vtstq_u32(vp5, m4), va5, vma5));
                c5 = vaddq_f32(c5, vbslq_f32(vtstq_u32(vp5, m5), va5, vma5));
                c6 = vaddq_f32(c6, vbslq_f32(vtstq_u32(vp5, m6), va5, vma5));
                c7 = vaddq_f32(c7, vbslq_f32(vtstq_u32(vp5, m7), va5, vma5));

                uint32x4_t vp6 = vdupq_n_u32(B6[j_int]);
                c0 = vaddq_f32(c0, vbslq_f32(vtstq_u32(vp6, m0), va6, vma6));
                c1 = vaddq_f32(c1, vbslq_f32(vtstq_u32(vp6, m1), va6, vma6));
                c2 = vaddq_f32(c2, vbslq_f32(vtstq_u32(vp6, m2), va6, vma6));
                c3 = vaddq_f32(c3, vbslq_f32(vtstq_u32(vp6, m3), va6, vma6));
                c4 = vaddq_f32(c4, vbslq_f32(vtstq_u32(vp6, m4), va6, vma6));
                c5 = vaddq_f32(c5, vbslq_f32(vtstq_u32(vp6, m5), va6, vma6));
                c6 = vaddq_f32(c6, vbslq_f32(vtstq_u32(vp6, m6), va6, vma6));
                c7 = vaddq_f32(c7, vbslq_f32(vtstq_u32(vp6, m7), va6, vma6));

                uint32x4_t vp7 = vdupq_n_u32(B7[j_int]);
                c0 = vaddq_f32(c0, vbslq_f32(vtstq_u32(vp7, m0), va7, vma7));
                c1 = vaddq_f32(c1, vbslq_f32(vtstq_u32(vp7, m1), va7, vma7));
                c2 = vaddq_f32(c2, vbslq_f32(vtstq_u32(vp7, m2), va7, vma7));
                c3 = vaddq_f32(c3, vbslq_f32(vtstq_u32(vp7, m3), va7, vma7));
                c4 = vaddq_f32(c4, vbslq_f32(vtstq_u32(vp7, m4), va7, vma7));
                c5 = vaddq_f32(c5, vbslq_f32(vtstq_u32(vp7, m5), va7, vma7));
                c6 = vaddq_f32(c6, vbslq_f32(vtstq_u32(vp7, m6), va7, vma7));
                c7 = vaddq_f32(c7, vbslq_f32(vtstq_u32(vp7, m7), va7, vma7));

                vst1q_f32(C_row + j_int * 32 + 0, c0);
                vst1q_f32(C_row + j_int * 32 + 4, c1);
                vst1q_f32(C_row + j_int * 32 + 8, c2);
                vst1q_f32(C_row + j_int * 32 + 12, c3);
                vst1q_f32(C_row + j_int * 32 + 16, c4);
                vst1q_f32(C_row + j_int * 32 + 20, c5);
                vst1q_f32(C_row + j_int * 32 + 24, c6);
                vst1q_f32(C_row + j_int * 32 + 28, c7);
            }
        }
    }
}
