#pragma once
#include <cstdint>
#include <cstddef>

// Optimized Matmul using register blocking/tiling and unrolling for SIMD/CSEL.
// We use an i-p-c-k loop order for maximum cache efficiency.
// We unroll the p-loop (rows of B) and the c-loop (columns of B) to increase 
// the reuse of a_val and a_neg values in registers and to increase instruction level parallelism.
// We further unroll the k-loop (32 bits) to provide a branchless sequence 
// that the compiler can transform into NEON VBSL or CSEL instructions.

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

            // We unroll the c-loop to process 4 chunks of 32 bits at once.
            // This allows the compiler to potentially load 4 packed integers 
            // and use them to update 128 floats.
            size_t c = 0;
            for (; c + 3 < K_ints; c += 4) {
                const uint32_t p0 = b_row[c];
                const uint32_t p1 = b_row[c + 1];
                const uint32_t p2 = b_row[c + 2];
                const uint32_t p3 = b_row[c + 3];

                float* c0 = &c_row[c * 32];
                float* c1 = &c_row[(c + 1) * 32];
                float* c2 = &c_row[(c + 2) * 32];
                float* c3 = &c_row[(c + 3) * 32];

                // Unroll 32 bits manually for all 4 chunks.
                // Using local variables for the packed bits to help the compiler.
                #define UPDATE_CHUNK(chunk, packed) \
                chunk[0]  += (packed & 0x00000001U) ? a_val : a_neg; \
                chunk[1]  += (packed & 0x00000002U) ? a_val : a_neg; \
                chunk[2]  += (packed & 0x00000004U) ? a_val : a_neg; \
                chunk[3]  += (packed & 0x00000008U) ? a_val : a_neg; \
                chunk[4]  += (packed & 0x00000010U) ? a_val : a_neg; \
                chunk[5]  += (packed & 0x00000020U) ? a_val : a_neg; \
                chunk[6]  += (packed & 0x00000040U) ? a_val : a_neg; \
                chunk[7]  += (packed & 0x00000080U) ? a_val : a_neg; \
                chunk[8]  += (packed & 0x00000100U) ? a_val : a_neg; \
                chunk[9]  += (packed & 0x00000200U) ? a_val : a_neg; \
                chunk[10] += (packed & 0x00000400U) ? a_val : a_neg; \
                chunk[11] += (packed & 0x00000800U) ? a_val : a_neg; \
                chunk[12] += (packed & 0x00001000U) ? a_val : a_neg; \
                chunk[13] += (packed & 0x00002000U) ? a_val : a_neg; \
                chunk[14] += (packed & 0x00004000U) ? a_val : a_neg; \
                chunk[15] += (packed & 0x00008000U) ? a_val : a_neg; \
                chunk[16] += (packed & 0x00010000U) ? a_val : a_neg; \
                chunk[17] += (packed & 0x00020000U) ? a_val : a_neg; \
                chunk[18] += (packed & 0x00040000U) ? a_val : a_neg; \
                chunk[19] += (packed & 0x00080000U) ? a_val : a_neg; \
                chunk[20] += (packed & 0x00100000U) ? a_val : a_neg; \
                chunk[21] += (packed & 0x00200000U) ? a_val : a_neg; \
                chunk[22] += (packed & 0x00400000U) ? a_val : a_neg; \
                chunk[23] += (packed & 0x00800000U) ? a_val : a_neg; \
                chunk[24] += (packed & 0x01000000U) ? a_val : a_neg; \
                chunk[25] += (packed & 0x02000000U) ? a_val : a_neg; \
                chunk[26] += (packed & 0x04000000U) ? a_val : a_neg; \
                chunk[27] += (packed & 0x08000000U) ? a_val : a_neg; \
                chunk[28] += (packed & 0x10000000U) ? a_val : a_neg; \
                chunk[29] += (packed & (1U << 29)) ? a_val : a_neg; \
                chunk[30] += (packed & (1U << 30)) ? a_val : a_neg; \
                chunk[31] += (packed & (1U << 31)) ? a_val : a_neg;

                UPDATE_CHUNK(c0, p0);
                UPDATE_CHUNK(c1, p1);
                UPDATE_CHUNK(c2, p2);
                UPDATE_CHUNK(c3, p3);
            }

            // Handle remaining c chunks
            for (; c < K_ints; ++c) {
                const uint32_t packed = b_row[c];
                float* c_chunk = &c_row[c * 32];
                #define UPDATE_SINGLE_CHUNK(chunk, packed) \
                chunk[0]  += (packed & 0x00000001U) ? a_val : a_neg; \
                chunk[1]  += (packed & 0x00000002U) ? a_val : a_neg; \
                chunk[2]  += (packed & 0x00000004U) ? a_val : a_neg; \
                chunk[3]  += (packed & 0x00000008U) ? a_val : a_neg; \
                chunk[4]  += (packed & 0x00000010U) ? a_val : a_neg; \
                chunk[5]  += (packed & 0x00000020U) ? a_val : a_neg; \
                chunk[6]  += (packed & 0x00000040U) ? a_val : a_neg; \
                chunk[7]  += (packed & 0x00000080U) ? a_val : a_neg; \
                chunk[8]  += (packed & 0x00000100U) ? a_val : a_neg; \
                chunk[9]  += (packed & 0x00000200U) ? a_val : a_neg; \
                chunk[10] += (packed & 0x00000400U) ? a_val : a_neg; \
                chunk[11] += (packed & 0x00000800U) ? a_val : a_neg; \
                chunk[12] += (packed & 0x00001000U) ? a_val : a_neg; \
                chunk[13] += (packed & 0x00002000U) ? a_val : a_neg; \
                chunk[14] += (packed & 0x00004000U) ? a_val : a_neg; \
                chunk[15] += (packed & 0x00008000U) ? a_val : a_neg; \
                chunk[16] += (packed & 0x00010000U) ? a_val : a_neg; \
                chunk[17] += (packed & 0x00020000U) ? a_val : a_neg; \
                chunk[18] += (packed & 0x00040000U) ? a_val : a_neg; \
                chunk[19] += (packed & 0x00080000U) ? a_val : a_neg; \
                chunk[20] += (packed & 0x00100000U) ? a_val : a_neg; \
                chunk[21] += (packed & 0x00200000U) ? a_val : a_neg; \
                chunk[22] += (packed & 0x00400000U) ? a_val : a_neg; \
                chunk[23] += (packed & 0x00800000U) ? a_val : a_neg; \
                chunk[24] += (packed & 0x01000000U) ? a_val : a_neg; \
                chunk[25] += (packed & 0x02000000U) ? a_val : a_neg; \
                chunk[26] += (packed & 0x04000000U) ? a_val : a_neg; \
                chunk[27] += (packed & 0x08000000U) ? a_val : a_neg; \
                chunk[28] += (packed & 0x10000000U) ? a_val : a_neg; \
                chunk[29] += (packed & (1U << 29)) ? a_val : a_neg; \
                chunk[30] += (packed & (1U << 30)) ? a_val : a_neg; \
                chunk[31] += (packed & (1U << 31)) ? a_val : a_neg;

                UPDATE_SINGLE_CHUNK(c_chunk, packed);
            }
            #undef UPDATE_CHUNK
            #undef UPDATE_SINGLE_CHUNK
        }
    }
}
