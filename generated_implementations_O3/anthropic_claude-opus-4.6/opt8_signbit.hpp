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

    // Strategy: for each bit in B, if bit=1 sign=+1, if bit=0 sign=-1.
    // So the value to add is a_val * (bit ? 1 : -1).
    // We can flip the sign bit of a_val: XOR a_val's bits with 0x80000000 when bit=0.
    // Equivalently, XOR with ((~bit) << 31).
    // 
    // Using NEON: expand each bit to a full 32-bit mask for the sign bit.
    // If bit=0, mask = 0x80000000 (flip sign). If bit=1, mask = 0 (keep sign).
    
    const uint32x4_t sign_bit = vdupq_n_u32(0x80000000u);

    for (size_t i = 0; i < M; ++i) {
        float* c_row = C + i * K;
        const float* a_row = A + i * K;
        
        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            uint32_t a_bits = *(const uint32_t*)&a_val;
            uint32x4_t va_bits = vdupq_n_u32(a_bits);
            
            const uint32_t* b_row = B + p * K_ints;
            
            for (size_t jb = 0; jb < K_ints; ++jb) {
                uint32_t packed = b_row[jb];
                float* c_ptr = c_row + jb * 32;
                
                // Process 4 bits at a time using NEON
                // For bits at positions 0,1,2,3:
                // Extract each bit, expand to mask, XOR with a_val sign bit if bit=0
                
                // Unrolled: 8 groups of 4 bits = 32 bits
                for (int g = 0; g < 32; g += 4) {
                    // Extract 4 bits
                    uint32_t b4 = (packed >> g);
                    
                    // For each bit: if 0, we need to flip sign of a_val
                    // mask = (bit == 0) ? 0x80000000 : 0
                    // ~bit & 1 gives 1 if bit=0, 0 if bit=1
                    // shift left by 31 gives sign mask
                    uint32_t m0 = ((~b4) & 1) << 31;
                    uint32_t m1 = ((~(b4 >> 1)) & 1) << 31;
                    uint32_t m2 = ((~(b4 >> 2)) & 1) << 31;
                    uint32_t m3 = ((~(b4 >> 3)) & 1) << 31;
                    
                    uint32_t marr[4] = {m0, m1, m2, m3};
                    uint32x4_t mask = vld1q_u32(marr);
                    
                    // XOR a_val bits with mask to flip sign when needed
                    uint32x4_t result_bits = veorq_u32(va_bits, mask);
                    float32x4_t to_add = vreinterpretq_f32_u32(result_bits);
                    
                    float32x4_t c_vec = vld1q_f32(c_ptr + g);
                    c_vec = vaddq_f32(c_vec, to_add);
                    vst1q_f32(c_ptr + g, c_vec);
                }
            }
        }
    }
}
