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
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            if (a_val == 0.0f) continue;
            
            const uint32_t* B_row = B + p * K_ints;
            float* C_row = C + i * K;
            
            float32x4_t v_a = vdupq_n_f32(a_val);
            float32x4_t v_ma = vdupq_n_f32(-a_val);

            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = B_row[j_int];
                uint32x4_t v_packed = vdupq_n_u32(packed);

                float32x4_t c0 = vld1q_f32(C_row + j_int * 32 + 0);
                float32x4_t c1 = vld1q_f32(C_row + j_int * 32 + 4);
                float32x4_t c2 = vld1q_f32(C_row + j_int * 32 + 8);
                float32x4_t c3 = vld1q_f32(C_row + j_int * 32 + 12);
                float32x4_t c4 = vld1q_f32(C_row + j_int * 32 + 16);
                float32x4_t c5 = vld1q_f32(C_row + j_int * 32 + 20);
                float32x4_t c6 = vld1q_f32(C_row + j_int * 32 + 24);
                float32x4_t c7 = vld1q_f32(C_row + j_int * 32 + 28);

                c0 = vaddq_f32(c0, vbslq_f32(vtstq_u32(v_packed, m0), v_a, v_ma));
                c1 = vaddq_f32(c1, vbslq_f32(vtstq_u32(v_packed, m1), v_a, v_ma));
                c2 = vaddq_f32(c2, vbslq_f32(vtstq_u32(v_packed, m2), v_a, v_ma));
                c3 = vaddq_f32(c3, vbslq_f32(vtstq_u32(v_packed, m3), v_a, v_ma));
                c4 = vaddq_f32(c4, vbslq_f32(vtstq_u32(v_packed, m4), v_a, v_ma));
                c5 = vaddq_f32(c5, vbslq_f32(vtstq_u32(v_packed, m5), v_a, v_ma));
                c6 = vaddq_f32(c6, vbslq_f32(vtstq_u32(v_packed, m6), v_a, v_ma));
                c7 = vaddq_f32(c7, vbslq_f32(vtstq_u32(v_packed, m7), v_a, v_ma));

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
