#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K){
    const size_t K_ints = K/32;
    // Build lookup table of 256*8 floats (signs for each byte value)
    const std::array<float, 256*8> sign_table = []{
        std::array<float, 256*8> t{};
        for(int v=0; v<256; ++v){
            for(int k=0; k<8; ++k){
                t[v*8 + k] = ((v >> k) & 1U) ? 1.0f : -1.0f;
            }
        }
        return t;
    }();

    for(size_t i=0; i<M; ++i){
        const float* a_row = A + i*K;
        float* rowC = C + i*K;
        std::memset(rowC, 0, sizeof(float)*K);

        for(size_t p=0; p<K; ++p){
            float a_val = a_row[p];
            float32x4_t a_vec = vdupq_n_f32(a_val);
            const uint32_t* packed_row = B + p*K_ints;

            for(size_t blk=0; blk<K_ints; ++blk){
                uint32_t bits = packed_row[blk];
                float* c_ptr = rowC + blk*32;
                // Process four bytes (8 cols each): total 32 cols
                for(size_t byte=0; byte<4; ++byte){
                    uint8_t sub = (bits >> (byte*8)) & 0xFFu;
                    const float* s = sign_table.data() + sub*8;
                    float32x4_t s_low = vld1q_f32(s);     // signs for first 4 cols
                    float32x4_t s_high = vld1q_f32(s+4);  // signs for next 4 cols
                    float32x4_t c0 = vld1q_f32(c_ptr + byte*8 + 0);
                    float32x4_t c1 = vld1q_f32(c_ptr + byte*8 + 4);
                    c0 = vfmaq_f32(c0, a_vec, s_low);
                    c1 = vfmaq_f32(c1, a_vec, s_high);
                    vst1q_f32(c_ptr + byte*8 + 0, c0);
                    vst1q_f32(c_ptr + byte*8 + 4, c1);
                }
            }
        }
    }
}
