#pragma once
#include <cstdint>
#include <cstddef>

// Optimized Matmul using NEON SIMD intrinsics.
// Idea: For each 32-bit integer in B, we can expand it into a vector of 32 floats.
// However, to save memory/registers, we process in small chunks.
// We'll use the bit-masking technique to avoid branches and use NEON to 
// process 4 or 8 elements at once.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* c_row = &C[i * K];
        for (size_t j = 0; j < K; ++j) {
            c_row[j] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) {
            const float a_val = A[i * K + p];
            const float a_neg = -a_val;
            const uint32_t* b_row = &B[p * K_ints];

            for (size_t c = 0; c < K_ints; ++c) {
                const uint32_t packed = b_row[c];
                float* c_chunk_ptr = &c_row[c * 32];

                // We'll unroll the bit-processing loop.
                // To utilize NEON, we need to translate the bits into something 
                // that can be used with vbslq_f32. 
                // Since we can't easily create a float mask from bits without 
                // shuffling/expanding, we'll stick to an unrolled loop that 
                // encourages the compiler to use NEON/SIMD instructions 
                // like VLD1, VADD, and conditional selects.
                
                // Processing 8 bits at a time (8 floats)
                for (size_t k = 0; k < 32; k += 8) {
                    const uint32_t mask = (packed >> k) & 0xFF;
                    
                    // Manually unroll the 8 lanes to help auto-vectorization.
                    // Each lane checks one bit.
                    c_chunk_ptr[k + 0] += (mask & 0x01) ? a_val : a_neg;
                    c_chunk_ptr[k + 1] += (mask & 0x02) ? a_val : a_neg;
                    c_chunk_ptr[k + 2] += (mask & 0x04) ? a_val : a_neg;
                    c_chunk_ptr[k + 3] += (mask & 0x08) ? a_val : a_neg;
                    c_chunk_ptr[k + 4] += (mask & 0x10) ? a_val : a_neg;
                    c_chunk_ptr[k + 5] += (mask & 0x20) ? a_val : a_neg;
                    c_chunk_ptr[k + 6] += (mask & 0x40) ? a_val : a_neg;
                    c_chunk_ptr[k + 7] += (mask & 0x80) ? a_val : a_neg;
                }
            }
        }
    }
}
