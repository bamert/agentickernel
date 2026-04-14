#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Zero C using NEON
    {
        float32x4_t zero = vdupq_n_f32(0.0f);
        for (size_t idx = 0; idx < M * K; idx += 16) {
            vst1q_f32(C + idx, zero);
            vst1q_f32(C + idx + 4, zero);
            vst1q_f32(C + idx + 8, zero);
            vst1q_f32(C + idx + 12, zero);
        }
    }

    // Key approach: for each p, process the packed B row.
    // For each uint32 in B row, expand 32 bits to sign floats and accumulate.
    // Use NEON sign-bit XOR: reinterpret a_val as uint32, XOR with sign_mask based on bit.
    // If bit=1 -> sign=+1 -> no flip -> XOR with 0
    // If bit=0 -> sign=-1 -> flip -> XOR with 0x80000000
    // sign_mask = (~bit & 1) << 31
    
    // Precompute lookup: for each 4-bit nibble (0-15), store 4 sign masks
    uint32_t sign_lut[16 * 4];
    for (int nibble = 0; nibble < 16; ++nibble) {
        for (int b = 0; b < 4; ++b) {
            int bit = (nibble >> b) & 1;
            sign_lut[nibble * 4 + b] = bit ? 0u : 0x80000000u;
        }
    }

    for (size_t i = 0; i < M; ++i) {
        float* __restrict__ c_row = C + i * K;
        const float* __restrict__ a_row = A + i * K;
        
        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            uint32_t a_bits;
            __builtin_memcpy(&a_bits, &a_val, 4);
            uint32x4_t va_bits = vdupq_n_u32(a_bits);
            
            const uint32_t* b_row = B + p * K_ints;
            
            for (size_t jb = 0; jb < K_ints; ++jb) {
                uint32_t packed = b_row[jb];
                float* c_ptr = c_row + jb * 32;
                
                // Process 32 bits using nibble LUT: 8 nibbles per uint32
                for (int n = 0; n < 8; ++n) {
                    int nibble = (packed >> (n * 4)) & 0xF;
                    uint32x4_t mask = vld1q_u32(&sign_lut[nibble * 4]);
                    uint32x4_t signed_a = veorq_u32(va_bits, mask);
                    float32x4_t to_add = vreinterpretq_f32_u32(signed_a);
                    float32x4_t c_vec = vld1q_f32(c_ptr + n * 4);
                    c_vec = vaddq_f32(c_vec, to_add);
                    vst1q_f32(c_ptr + n * 4, c_vec);
                }
            }
        }
    }
}
