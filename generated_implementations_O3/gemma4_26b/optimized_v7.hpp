#pragma once
#include <cstdint>
#include <cstddef>

// Optimized Matmul using NEON-friendly loop structure and bitwise logic.
// We use the i-p-c-k loop order for maximum cache efficiency.
// We process 4 chunks of B (128 bits) and use a bit-manipulation trick to
// reduce branchy dependency. 
// To avoid branches, we can use the fact that (bit ? 1 : -1) is 2*bit - 1.
// But with floats, we'll instead use the CSEL-friendly ternary.
// We also ensure we are accessing C row elements in a way that allows
// the compiler to utilize SIMD lanes.

void matmul(const float* A, const uint3int32_t* B, float* C, size_t M, size_t K) {
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

                // Unrolling the k-loop to 8 bits to encourage NEONization
                // and reduce loop overhead.
                for (size_t k = 0; k < 32; k += 8) {
                    // We use individual bits to allow the compiler to transform
                    // these into conditional select (CSEL) or bitwise masking.
                    c_chunk[k + 0] += (packed & (1U << 0)) ? a_val : a_neg;
                    c_chunk[k + 1] += (packed & (1U << 1)) ? a_val : a_neg;
                    c_chunk[k + 2] += (packed & (1U << 2)) ? a_val : a_neg;
                    c_chunk[k + 3] += (packed & (1U << 3)) ? a_val : a_neg;
                    c_chunk[k + 4] += (packed & (1U << 4)) ? a_val : a_neg;
                    c_chunk[k + 5] += (packed & (1U << 5)) ? a_val : a_neg;
                    c_chunk[k + 6] += (packed & (1U << 6)) ? a_val : a_neg;
                    c_chunk[k + 7] += (packed & (1U << 7)) ? a_val : a_neg;

                    // We shift the packed value to handle the next group of 8 bits
                    // or simply use explicit offsets.
                    c_chunk[k + 8]  += (packed & (1U << 8))  ? a_val : a_neg;
                    c_chunk[k + 9]  += (packed & (1U << 9))  ? a_val : a_neg;
                    c_chunk[k + 10] += (packed & (1U << 10)) ? a_val : a_neg;
                    c_chunk[k + 11] += (packed & (1U << 11)) ? a_val : a_neg;
                    c_chunk[k + 12] += (packed & (1U << 12)) ? a_val : a_neg;
                    c_chunk[k + 13] += (packed & (1U << 13)) ? a_val : a_neg;
                    c_chunk[k + 14] += (packed & (1U << 14)) ? a_val : a_neg;
                    c_chunk[k + 15] += (packed & (1U << 15)) ? a_val : a_neg;

                    c_chunk[k + 16] += (packed & (1U << 16)) ? a_val : a_neg;
                    c_chunk[k + 17] += (packed & (1U << 17)) ? a_val : a_neg;
                    c_chunk[k + 18] += (packed & (1U << 18)) ? a_val : a_neg;
                    c_chunk[k + 19] += (packed & (1U << 19)) ? a_val : a_neg;
                    c_chunk[k + 20] += (packed & (1U << 20)) ? a_val : a_neg;
                    c_chunk[k + 21] += (packed & (1U << 21)) ? a_val : a_neg;
                    c_chunk[k + 22] += (packed & (1U << 22)) ? a_val : a_neg;
                    c_chunk[k + 23] += (packed & (1U << 23)) ? a_val : a_neg;

                    c_chunk[k + 24] += (packed & (1U << 24)) ? a_val : a_neg;
                    c_chunk[k + 25] += (packed & (1U << 25)) ? a_val : a_neg;
                    c_chunk[k + 26] += (packed & (1U << 26)) ? a_val : a_neg;
                    c_chunk[k + 27] += (packed & (1U << 27)) ? a_val : a_neg;
                    c_chunk[k + 28] += (packed & (1U << 28)) ? a_val : a_neg;
                    c_chunk[k + 29] += (packed & (1U << 29)) ? a_val : a_neg;
                    c_chunk[k + 30] += (packed & (1U << 30)) ? a_val : a_neg;
                    c_chunk[k + 31] += (packed & (1U << 31)) ? a_val : a_neg;
                    
                    break; // We actually processed the whole 32 bits. 
                }
            }
        }
    }
}
