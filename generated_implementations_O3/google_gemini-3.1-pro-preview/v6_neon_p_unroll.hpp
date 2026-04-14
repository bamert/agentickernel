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
        for (size_t p = 0; p < K; p += 4) {
            float a0 = A[i * K + p + 0];
            float a1 = A[i * K + p + 1];
            float a2 = A[i * K + p + 2];
            float a3 = A[i * K + p + 3];

            float32x4_t va0 = vdupq_n_f32(a0), vma0 = vdupq_n_f32(-a0);
            float32x4_t va1 = vdupq_n_f32(a1), vma1 = vdupq_n_f32(-a1);
            float32x4_t va2 = vdupq_n_f32(a2), vma2 = vdupq_n_f32(-a2);
            float32x4_t va3 = vdupq_n_f32(a3), vma3 = vdupq_n_f32(-a3);

            const uint32_t* B_row0 = B + (p + 0) * K_ints;
            const uint32_t* B_row1 = B + (p + 1) * K_ints;
            const uint32_t* B_row2 = B + (p + 2) * K_ints;
            const uint32_t* B_row3 = B + (p + 3) * K_ints;
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

                // p = 0
                uint32x4_t v_packed0 = vdupq_n_u32(B_row0[j_int]);
                c0 = vaddq_f32(c0, vbslq_f32(vtstq_u32(v_packed0, m0), va0, vma0));
                c1 = vaddq_f32(c1, vbslq_f32(vtstq_u32(v_packed0, m1), va0, vma0));
                c2 = vaddq_f32(c2, vbslq_f32(vtstq_u32(v_packed0, m2), va0, vma0));
                c3 = vaddq_f32(c3, vbslq_f32(vtstq_u32(v_packed0, m3), va0, vma0));
                c4 = vaddq_f32(c4, vbslq_f32(vtstq_u32(v_packed0, m4), va0, vma0));
                c5 = vaddq_f32(c5, vbslq_f32(vtstq_u32(v_packed0, m5), va0, vma0));
                c6 = vaddq_f32(c6, vbslq_f32(vtstq_u32(v_packed0, m6), va0, vma0));
                c7 = vaddq_f32(c7, vbslq_f32(vtstq_u32(v_packed0, m7), va0, vma0));

                // p = 1
                uint32x4_t v_packed1 = vdupq_n_u32(B_row1[j_int]);
                c0 = vaddq_f32(c0, vbslq_f32(vtstq_u32(v_packed1, m0), va1, vma1));
                c1 = vaddq_f32(c1, vbslq_f32(vtstq_u32(v_packed1, m1), va1, vma1));
                c2 = vaddq_f32(c2, vbslq_f32(vtstq_u32(v_packed1, m2), va1, vma1));
                c3 = vaddq_f32(c3, vbslq_f32(vtstq_u32(v_packed1, m3), va1, vma1));
                c4 = vaddq_f32(c4, vbslq_f32(vtstq_u32(v_packed1, m4), va1, vma1));
                c5 = vaddq_f32(c5, vbslq_f32(vtstq_u32(v_packed1, m5), va1, vma1));
                c6 = vaddq_f32(c6, vbslq_f32(vtstq_u32(v_packed1, m6), va1, vma1));
                c7 = vaddq_f32(c7, vbslq_f32(vtstq_u32(v_packed1, m7), va1, vma1));

                // p = 2
                uint32x4_t v_packed2 = vdupq_n_u32(B_row2[j_int]);
                c0 = vaddq_f32(c0, vbslq_f32(vtstq_u32(v_packed2, m0), va2, vma2));
                c1 = vaddq_f32(c1, vbslq_f32(vtstq_u32(v_packed2, m1), va2, vma2));
                c2 = vaddq_f32(c2, vbslq_f32(vtstq_u32(v_packed2, m2), va2, vma2));
                c3 = vaddq_f32(c3, vbslq_f32(vtstq_u32(v_packed2, m3), va2, vma2));
                c4 = vaddq_f32(c4, vbslq_f32(vtstq_u32(v_packed2, m4), va2, vma2));
                c5 = vaddq_f32(c5, vbslq_f32(vtstq_u32(v_packed2, m5), va2, vma2));
                c6 = vaddq_f32(c6, vbslq_f32(vtstq_u32(v_packed2, m6), va2, vma2));
                c7 = vaddq_f32(c7, vbslq_f32(vtstq_u32(v_packed2, m7), va2, vma2));

                // p = 3
                uint32x4_t v_packed3 = vdupq_n_u32(B_row3[j_int]);
                c0 = vaddq_f32(c0, vbslq_f32(vtstq_u32(v_packed3, m0), va3, vma3));
                c1 = vaddq_f32(c1, vbslq_f32(vtstq_u32(v_packed3, m1), va3, vma3));
                c2 = vaddq_f32(c2, vbslq_f32(vtstq_u32(v_packed3, m2), va3, vma3));
                c3 = vaddq_f32(c3, vbslq_f32(vtstq_u32(v_packed3, m3), va3, vma3));
                c4 = vaddq_f32(c4, vbslq_f32(vtstq_u32(v_packed3, m4), va3, vma3));
                c5 = vaddq_f32(c5, vbslq_f32(vtstq_u32(v_packed3, m5), va3, vma3));
                c6 = vaddq_f32(c6, vbslq_f32(vtstq_u32(v_packed3, m6), va3, vma3));
                c7 = vaddq_f32(c7, vbslq_f32(vtstq_u32(v_packed3, m7), va3, vma3));

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
