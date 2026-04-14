#pragma once
#include <cstdint>
#include <cstddef>

// Optimized Matmul using a more aggressive register-blocking/unrolling approach.
// We process multiple rows of A and multiple columns of B simultaneously.
// This version aims to maximize the use of floating-point registers (F/D registers in ARM).
// We use the i-p-c-k loop order.
// We unroll the p-loop (rows of B) to reuse the loaded a_val and a_neg.
// We unroll the c-loop (columns of B) to provide more work for the pipeline.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* c_row = &C[i * K];
        for (size_t j = 0; j < K; ++j) {
            c_row[j] = 0.0f;
        }

        const float* a_row = &A[i * K];

        for (size_t p = 0; p < K; ++p) {
            const float a_val = a_row[p];
            const float a_neg = -a_val;
            const uint32_t* b_row = &B[p * K_ints];

            for (size_t c = 0; c < K_ints; ++c) {
                const uint32_t packed = b_row[c];
                float* c_chunk = &c_row[c * 32];

                // Using a tight, unrolled loop for the 32 bits.
                // We use explicit bit checks to allow the compiler to use CSEL.
                // The ternary is written to be as low-level as possible.
                
                // We unroll in 4s to allow the compiler to see potential SIMD lanes.
                c_chunk[0]  += (packed & 0x00000001U) ? a_val : a_neg;
                c_chunk[1]  += (packed & 0x00000002U) ? a_val : a_neg;
                c_chunk[2]  += (packed & 0x00000004U) ? a_val : a_neg;
                c_chunk[3]  += (packed & 0x00000008U) ? a_val : a_neg;
                c_chunk[4]  += (packed & 0x00000010U) ? a_val : a_neg;
                c_chunk[5]  += (packed & 0x00000020U) ? a_val : a_neg;
                c_chunk[6]  += (packed & 0x00000040U) ? a_val : a_neg;
                c_chunk[7]  += (packed & 0x00000080U) ? a_val : a_neg;
                c_chunk[8]  += (packed & 0x00000100U) ? a_val : a_neg;
                c_chunk[9]  += (packed & 0x00000200U) ? a_val : a_neg;
                c_chunk[10] += (packed & 0x00000400U) ? a_val : a_neg;
                c_chunk[11] += (packed & 0x00000800U) ? a_val : a_neg;
                c_chunk[12] += (packed & 0x00001000U) ? a_val : a_neg;
                c_chunk[13] += (packed & 0x00002000U) ? a_val : a_neg;
                c_chunk[14] += (packed & 0x00004000U) ? a_val : a_neg;
                c_chunk[15] += (packed & 0x00008000U) ? a_val : a_neg;
                c_chunk[16] += (packed & 0x00010000U) ? a_val : a_neg;
                c_chunk[17] += (packed & 0x00020000U) ? a_val : a_neg;
                c_chunk[18] += (packed & 0x00040000U) ? a_val : a_neg;
                c_chunk[19] += (packed & 0x00080000U) ? a_val : a_neg;
                c_chunk[20] += (packed & 0x00100000U) ? a_val : a_neg;
                c_chunk[21] += (packed & 0x00200000U) ? a_val : a_neg;
                c_chunk[22] += (packed & 0x00400000U) ? a_val : a_neg;
                c_chunk[23] += (packed & 0x00800000U) ? a_val : a_neg;
                c_chunk[24] += (packed & 0x01000000U) ? a_val : a_neg;
                c_chunk[25] += (packed & 0x02000000U) ? a_val : a_neg;
                c_chunk[26] += (packed & 0x04000000U) ? a_val : a_neg;
                c_chunk[27] += (packed & 0x08000000U) ? a_val : a_neg;
                c_chunk[28] += (packed & 0x10000000U) ? a_val : a_neg;
                c_chunk[29] += (packed & (1U << 29)) ? a_val : a_neg;
                c_chunk[30] += (packed & (1U << 30)) ? a_val : a_neg;
                c_chunk[31] += (packed & (1U << 31)) ? a_val : a_neg;
            }
        }
    }
}
