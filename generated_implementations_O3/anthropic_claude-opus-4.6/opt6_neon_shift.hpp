#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Zero out C
    for (size_t idx = 0; idx + 4 <= M * K; idx += 4) {
        vst1q_f32(C + idx, vdupq_n_f32(0.0f));
    }

    // Precompute shift amounts for extracting bits
    // For bits 0,1,2,3 we shift right by 0,1,2,3 then mask with 1
    // Then convert: bit -> (bit << 1) - 1 gives sign as int, then convert to float
    // Or: reinterpret bit as float after shifting into sign bit position
    
    // Better: use the fact that float representation of 1.0 = 0x3F800000
    // and -1.0 = 0xBF800000. The only difference is the sign bit (bit 31).
    // So: sign_float = 1.0f with bit 31 set to (1-bit) = ~bit
    
    const uint32x4_t one_u = vdupq_n_u32(1);
    const float32x4_t two_f = vdupq_n_f32(2.0f);
    const float32x4_t one_f = vdupq_n_f32(1.0f);

    for (size_t i = 0; i < M; ++i) {
        float* c_row = C + i * K;
        const float* a_row = A + i * K;
        
        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            float32x4_t va = vdupq_n_f32(a_val);
            float32x4_t neg_va = vnegq_f32(va);
            
            const uint32_t* b_row = B + p * K_ints;
            
            for (size_t jb = 0; jb < K_ints; ++jb) {
                uint32_t packed = b_row[jb];
                float* c_ptr = c_row + jb * 32;
                
                // Use NEON to expand bits to masks
                // Broadcast packed to all 4 lanes
                uint32x4_t vpacked = vdupq_n_u32(packed);
                
                // For bits 0-3: shift right by {0,1,2,3}, mask with 1, compare
                {
                    int32x4_t shifts = {0, -1, -2, -3};
                    uint32x4_t shifted = vshlq_u32(vpacked, shifts);
                    uint32x4_t bits = vandq_u32(shifted, one_u);
                    // bits is 0 or 1. Convert to mask: compare with 1
                    uint32x4_t mask = vceqq_u32(bits, one_u);
                    float32x4_t selected = vbslq_f32(mask, va, neg_va);
                    float32x4_t c_vec = vld1q_f32(c_ptr);
                    vst1q_f32(c_ptr, vaddq_f32(c_vec, selected));
                }
                {
                    int32x4_t shifts = {-4, -5, -6, -7};
                    uint32x4_t shifted = vshlq_u32(vpacked, shifts);
                    uint32x4_t bits = vandq_u32(shifted, one_u);
                    uint32x4_t mask = vceqq_u32(bits, one_u);
                    float32x4_t selected = vbslq_f32(mask, va, neg_va);
                    float32x4_t c_vec = vld1q_f32(c_ptr + 4);
                    vst1q_f32(c_ptr + 4, vaddq_f32(c_vec, selected));
                }
                {
                    int32x4_t shifts = {-8, -9, -10, -11};
                    uint32x4_t shifted = vshlq_u32(vpacked, shifts);
                    uint32x4_t bits = vandq_u32(shifted, one_u);
                    uint32x4_t mask = vceqq_u32(bits, one_u);
                    float32x4_t selected = vbslq_f32(mask, va, neg_va);
                    float32x4_t c_vec = vld1q_f32(c_ptr + 8);
                    vst1q_f32(c_ptr + 8, vaddq_f32(c_vec, selected));
                }
                {
                    int32x4_t shifts = {-12, -13, -14, -15};
                    uint32x4_t shifted = vshlq_u32(vpacked, shifts);
                    uint32x4_t bits = vandq_u32(shifted, one_u);
                    uint32x4_t mask = vceqq_u32(bits, one_u);
                    float32x4_t selected = vbslq_f32(mask, va, neg_va);
                    float32x4_t c_vec = vld1q_f32(c_ptr + 12);
                    vst1q_f32(c_ptr + 12, vaddq_f32(c_vec, selected));
                }
                {
                    int32x4_t shifts = {-16, -17, -18, -19};
                    uint32x4_t shifted = vshlq_u32(vpacked, shifts);
                    uint32x4_t bits = vandq_u32(shifted, one_u);
                    uint32x4_t mask = vceqq_u32(bits, one_u);
                    float32x4_t selected = vbslq_f32(mask, va, neg_va);
                    float32x4_t c_vec = vld1q_f32(c_ptr + 16);
                    vst1q_f32(c_ptr + 16, vaddq_f32(c_vec, selected));
                }
                {
                    int32x4_t shifts = {-20, -21, -22, -23};
                    uint32x4_t shifted = vshlq_u32(vpacked, shifts);
                    uint32x4_t bits = vandq_u32(shifted, one_u);
                    uint32x4_t mask = vceqq_u32(bits, one_u);
                    float32x4_t selected = vbslq_f32(mask, va, neg_va);
                    float32x4_t c_vec = vld1q_f32(c_ptr + 20);
                    vst1q_f32(c_ptr + 20, vaddq_f32(c_vec, selected));
                }
                {
                    int32x4_t shifts = {-24, -25, -26, -27};
                    uint32x4_t shifted = vshlq_u32(vpacked, shifts);
                    uint32x4_t bits = vandq_u32(shifted, one_u);
                    uint32x4_t mask = vceqq_u32(bits, one_u);
                    float32x4_t selected = vbslq_f32(mask, va, neg_va);
                    float32x4_t c_vec = vld1q_f32(c_ptr + 24);
                    vst1q_f32(c_ptr + 24, vaddq_f32(c_vec, selected));
                }
                {
                    int32x4_t shifts = {-28, -29, -30, -31};
                    uint32x4_t shifted = vshlq_u32(vpacked, shifts);
                    uint32x4_t bits = vandq_u32(shifted, one_u);
                    uint32x4_t mask = vceqq_u32(bits, one_u);
                    float32x4_t selected = vbslq_f32(mask, va, neg_va);
                    float32x4_t c_vec = vld1q_f32(c_ptr + 28);
                    vst1q_f32(c_ptr + 28, vaddq_f32(c_vec, selected));
                }
            }
        }
    }
}
