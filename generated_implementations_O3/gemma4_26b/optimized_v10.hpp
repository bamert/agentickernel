#pragma once
#include <cstdint>
#include <cstddef>

// Optimized Matmul using NEON-friendly loop structure and full unrolling.
// The primary goal is to provide the compiler with a pattern that is 
// completely branchless and allows for SIMD vectorization.
// We use the i-p-c-k loop order for maximum cache efficiency.
// We process the bits of the uint32_t by manually unrolling the 32 bits.
// This avoids the overhead of the loop and the branches of the ternary operator,
// allowing the compiler to use instructions like VBSL (Vector Bitwise Select).

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* c_row = &C[i * K];
        // Initialize C row to zero.
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

                // Manually unrolling the 32 bits of the packed integer.
                // This approach is branchless and allows the compiler to 
                // use the CSEL or bitwise masking with NEON.
                // We use a sequence of additions with ternary operators 
                // which are extremely efficient on ARM (compiles to CSEL).
                
                c_chunk[0]  += (packed & (1U << 0))  ? a_val : a_neg;
                c_chunk[1]  += (packed & (1U << 1))  ? a_val : a_neg;
                c_chunk[2]  += (packed & (1U << 2))  ? a_val : a_neg;
                c_chunk[3]  += (packed & (1U << 3))  ? a_val : a_neg;
                c_chunk[4]  += (packed & (1U << 4))  ? a_val : a_neg;
                c_chunk[5]  += (packed & (1U << 5))  ? a_val : a_neg;
                c_chunk[6]  += (packed & (1U << 6))  ? a_val : a_neg;
                c_chunk[7]  += (packed & (1U << 7))  ? a_val : a_neg;
                c_chunk[8]  += (packed & (1U << 8))  ? a_val : a_neg;
                c_chunk[9]  += (packed & (1U << 9))  ? a_val : a_neg;
                c_chunk[10] += (packed & (1U << 10)) ? a_val : a_neg;
                c_chunk[11] += (packed & (1U << 11)) ? a_val : a_neg;
                c_chunk[12] += (packed & (1U << 12)) ? a_val : a_neg;
                c_chunk[13] += (packed & (1U << 13)) ? a_val : a_neg;
                c_chunk[14] += (packed & (1U << 14)) ? a_val : a_neg;
                c_chunk[15] += (packed & (1U << 15)) ? a_val : a_neg;
                c_chunk[16] += (packed & (1U << 16)) ? a_val : a_neg;
                c_chunk[17] += (packed & (1U << 17)) ? a_val : a_neg;
                c_chunk[18] += (packed & (1U << 18)) ? a_val : a_neg;
                c_chunk[19] += (packed & (1U << 19)) ? a_val : a_neg;
                c_chunk[20] += (packed & (1U << 20)) ? a_val : a_neg;
                c_chunk[21] += (packed & (1U << 21)) ? a_val : a_neg;
                c_chunk[22] += (packed & (1U << 22)) ? a_val : a_neg;
                c_chunk[23] += (packed & (1U << 23)) ? a_val : a_neg;
                c_chunk[24] += (packed & (1U << 24)) ? a_val : a_neg;
                c_chunk[25] += (packed & (1U << 25)) ? a_val : a_neg;
                c_chunk[26] += (packed & (1U << 26)) ? a_val : a_neg;
                c_chunk[27] += (packed & (1U << 27)) ? a_val : a_neg;
                c_chunk[28] += (packed & (1U << 28)) ? a_val : a_neg;
                c_chunk[29] += (packed & (1U << 29)) ? a_val : a_neg;
                c_chunk[30] += (packed & (1U << 30)) ? a_val : a_neg;
                c_chunk[31] += (packed & (1U << 31)) ? a_val : a_neg;
            }
        }
    }
}
