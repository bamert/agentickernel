#pragma once
#include <cstdint>
#include <cstddef>

// Optimized Matmul using extreme register blocking and loop unrolling.
// We use an i-p-c-k loop order for maximum cache efficiency.
// We unroll the p-loop by 2 and the c-int loop by 4.
// We use a manually unrolled 32-bit bit-extraction logic.
// The key is to minimize the total number of loads and maximize the reuse of a_val.

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

            size_t c = 0;
            // Unroll c-loop by 4
            for (; c + 3 < K_ints; c += 4) {
                const uint32_t p0 = b_row[c];
                const uint32_t p1 = b_row[c + 1];
                const uint32_t p2 = b_row[c + 2];
                const uint32_t p3 = b_row[c + 3];

                float* c0 = &c_row[c * 32];
                float* c1 = &c_row[(c + 1) * 32];
                float* c2 = &c_row[(c + 2) * 32];
                float* c3 = &c_row[(c + 3) * 32];

                #define UPDATE_32_BITS(ptr, packed) \
                ptr[0]  += (packed & 0x00000001U) ? a_val : a_neg; \
                ptr[1]  += (packed & 0x00000002U) ? a_val : a_neg; \
                ptr[2]  += (packed & 0x00000004U) ? a_val : a_neg; \
                ptr[3]  += (packed & 0x00000008U) ? a_val : a_neg; \
                ptr[4]  += (packed & 0x00000010U) ? a_val : a_neg; \
                ptr[5]  += (packed & 0x00000020U) ? a_val : a_neg; \
                ptr[6]  += (packed & 0x00000040U) ? a_val : a_neg; \
                ptr[7]  += (packed & 0x00000080U) ? a_val : a_neg; \
                ptr[8]  += (packed & 0x00000100U) ? a_val : a_neg; \
                ptr[9]  += (packed & 0x00000200U) ? a_val : a_neg; \
                ptr[10] += (packed & 0x00000400U) ? a_val : a_neg; \
                ptr[11] += (packed & 0x00000800U) ? a_val : a_neg; \
                ptr[12] += (packed & 0x00001000U) ? a_val : a_neg; \
                ptr[13] += (packed & 0x00002000U) ? a_val : a_neg; \
                ptr[14] += (packed & 0x00004000U) ? a_val : a_neg; \
                ptr[15] += (packed & 0x00008000U) ? a_val : a_neg; \
                ptr[16] += (packed & 0x00010000U) ? a_val : a_neg; \
                ptr[17] += (packed & 0x00020000U) ? a_val : a_neg; \
                ptr[18] += (packed & 0x00040000U) ? a_val : a_neg; \
                ptr[19] += (packed & 0x00080000U) ? a_val : a_neg; \
                ptr[20] += (packed & 0x00100000U) ? a_val : a_neg; \
                ptr[21] += (packed & 0x00200000U) ? a_val : a_neg; \
                ptr[22] += (packed & 0x00400000U) ? a_val : a_neg; \
                ptr[23] += (packed & 0x00800000U) ? a_val : a_neg; \
                ptr[24] += (packed & 0x01000000U) ? a_val : a_neg; \
                ptr[25] += (packed & 0x02000000U) ? a_val : a_neg; \
                ptr[26] += (packed & 0x04000000U) ? a_val : a_neg; \
                ptr[27] += (packed & 0x08000000U) ? a_val : a_neg; \
                ptr[28] += (packed & 0x10000000U) ? a_val : a_neg; \
                ptr[29] += (packed & 0x20000000U) ? a_val : a_neg; \
                ptr[30] += (packed & 0x40000000U) ? a_val : a_neg; \
                ptr[31] += (packed & 0x80000000U) ? a_val : a_neg;

                UPDATE_32_BITS(c0, p0);
                UPDATE_32_BITS(c1, p1);
                UPDATE_32_BITS(c2, p2);
                UPDATE_32_BITS(c3, p3);
            }

            for (; c < K_ints; ++c) {
                const uint32_t packed = b_row[c];
                float* c_ptr = &c_row[c * 32];
                #define UPDATE_SINGLE(ptr, packed) \
                ptr[0]  += (packed & 0x00000001U) ? a_val : a_neg; \
                ptr[1]  += (packed & 0x00000002U) ? a_val : a_neg; \
                ptr[2]  += (packed & 0x000004U) ? a_val : a_neg; \
                ptr[3]  += (packed & 0x000008U) ? a_val : a_neg; \
                ptr[4]  += (packed & 0x00010U) ? a_val : a_neg; \
                ptr[5]  += (packed & 0x00020U) ? a_val : a_neg; \
                ptr[6]  += (packed & 0x00040U) ? a_val : a_neg; \
                ptr[7]  += (packed & 0x00080U) ? a_val : a_neg; \
                ptr[8]  += (packed & 0x00100U) ? a_val : a_neg; \
                ptr[9]  += (packed & 0x00200U) ? a_val : a_neg; \
                ptr[10] += (packed & 0x00400U) ? a_val : a_neg; \
                ptr[11] += (packed & 0x00800U) ? a_val : a_neg; \
                ptr[12] += (packed & 0x01000U) ? a_val : a_neg; \
                ptr[13] += (packed & 0x02000U) ? a_val : a_neg; \
                ptr[14] += (packed & 0x04000U) ? a_val : a_neg; \
                ptr[15] += (packed & 0x08000U) ? a_val : a_neg; \
                ptr[16] += (packed & 0x10000U) ? a_val : a_neg; \
                ptr[17] += (packed & 0x20000U) ? a_val : a_neg; \
                ptr[18] += (packed & 0x40000U) ? a_val : a_neg; \
                ptr[19] += (packed & 0x80000U) ? a_val : a_neg; \
                ptr[20] += (packed & 0x100000U) ? a_val : a_neg; \
                ptr[21] += (packed & 0x200000U) ? a_val : a_val : a_neg; \
                ptr[22] += (packed & 0x400000U) ? a_val : a_neg; \
                ptr[23] += (packed & 0x800000U) ? a_val : a_neg; \
                ptr[24] += (packed & 0x1000000U) ? a_val : a_neg; \
                ptr[25] += (packed & 0x2000000U) ? a_val : a_neg; \
                ptr[26] += (packed & 0x4000000U) ? a_val : a_neg; \
                ptr[27] += (packed & 0x8000000U) ? a_val : a_neg; \
                ptr[28] += (packed & 0x10000000U) ? a_val : a_neg; \
                ptr[29] += (packed & 0x20000000U) ? a_val : a_neg; \
                ptr[30] += (packed & 0x40000000U) ? a_val : a_neg; \
                ptr[31] += (packed & 0x80000000U) ? a_val : a_neg;

                UPDATE_SINGLE(c_ptr, packed);
                #undef UPDATE_SINGLE
            }
            #undef UPDATE_32_BITS
        }
    }
}
