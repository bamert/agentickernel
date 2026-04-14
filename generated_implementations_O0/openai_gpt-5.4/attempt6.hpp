#pragma once

typedef unsigned int uint32_t;
typedef decltype(sizeof(0)) size_t;

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;
    for (size_t i = 0; i < M; ++i) {
        const float* arow = A + i * K;
        float* crow = C + i * K;
        for (size_t j = 0; j < K; ++j) crow[j] = 0.0f;

        for (size_t p = 0; p + 1 < K; p += 2) {
            const float a0 = arow[p];
            const float n0 = -a0;
            const float a1 = arow[p + 1];
            const float n1 = -a1;
            const uint32_t* b0 = B + p * K_ints;
            const uint32_t* b1 = b0 + K_ints;
            size_t j = 0;
            for (size_t w = 0; w < K_ints; ++w, j += 32) {
                const uint32_t bits0 = b0[w];
                const uint32_t bits1 = b1[w];
                crow[j + 0]  += ((bits0 & (1u << 0))  ? a0 : n0) + ((bits1 & (1u << 0))  ? a1 : n1);
                crow[j + 1]  += ((bits0 & (1u << 1))  ? a0 : n0) + ((bits1 & (1u << 1))  ? a1 : n1);
                crow[j + 2]  += ((bits0 & (1u << 2))  ? a0 : n0) + ((bits1 & (1u << 2))  ? a1 : n1);
                crow[j + 3]  += ((bits0 & (1u << 3))  ? a0 : n0) + ((bits1 & (1u << 3))  ? a1 : n1);
                crow[j + 4]  += ((bits0 & (1u << 4))  ? a0 : n0) + ((bits1 & (1u << 4))  ? a1 : n1);
                crow[j + 5]  += ((bits0 & (1u << 5))  ? a0 : n0) + ((bits1 & (1u << 5))  ? a1 : n1);
                crow[j + 6]  += ((bits0 & (1u << 6))  ? a0 : n0) + ((bits1 & (1u << 6))  ? a1 : n1);
                crow[j + 7]  += ((bits0 & (1u << 7))  ? a0 : n0) + ((bits1 & (1u << 7))  ? a1 : n1);
                crow[j + 8]  += ((bits0 & (1u << 8))  ? a0 : n0) + ((bits1 & (1u << 8))  ? a1 : n1);
                crow[j + 9]  += ((bits0 & (1u << 9))  ? a0 : n0) + ((bits1 & (1u << 9))  ? a1 : n1);
                crow[j + 10] += ((bits0 & (1u << 10)) ? a0 : n0) + ((bits1 & (1u << 10)) ? a1 : n1);
                crow[j + 11] += ((bits0 & (1u << 11)) ? a0 : n0) + ((bits1 & (1u << 11)) ? a1 : n1);
                crow[j + 12] += ((bits0 & (1u << 12)) ? a0 : n0) + ((bits1 & (1u << 12)) ? a1 : n1);
                crow[j + 13] += ((bits0 & (1u << 13)) ? a0 : n0) + ((bits1 & (1u << 13)) ? a1 : n1);
                crow[j + 14] += ((bits0 & (1u << 14)) ? a0 : n0) + ((bits1 & (1u << 14)) ? a1 : n1);
                crow[j + 15] += ((bits0 & (1u << 15)) ? a0 : n0) + ((bits1 & (1u << 15)) ? a1 : n1);
                crow[j + 16] += ((bits0 & (1u << 16)) ? a0 : n0) + ((bits1 & (1u << 16)) ? a1 : n1);
                crow[j + 17] += ((bits0 & (1u << 17)) ? a0 : n0) + ((bits1 & (1u << 17)) ? a1 : n1);
                crow[j + 18] += ((bits0 & (1u << 18)) ? a0 : n0) + ((bits1 & (1u << 18)) ? a1 : n1);
                crow[j + 19] += ((bits0 & (1u << 19)) ? a0 : n0) + ((bits1 & (1u << 19)) ? a1 : n1);
                crow[j + 20] += ((bits0 & (1u << 20)) ? a0 : n0) + ((bits1 & (1u << 20)) ? a1 : n1);
                crow[j + 21] += ((bits0 & (1u << 21)) ? a0 : n0) + ((bits1 & (1u << 21)) ? a1 : n1);
                crow[j + 22] += ((bits0 & (1u << 22)) ? a0 : n0) + ((bits1 & (1u << 22)) ? a1 : n1);
                crow[j + 23] += ((bits0 & (1u << 23)) ? a0 : n0) + ((bits1 & (1u << 23)) ? a1 : n1);
                crow[j + 24] += ((bits0 & (1u << 24)) ? a0 : n0) + ((bits1 & (1u << 24)) ? a1 : n1);
                crow[j + 25] += ((bits0 & (1u << 25)) ? a0 : n0) + ((bits1 & (1u << 25)) ? a1 : n1);
                crow[j + 26] += ((bits0 & (1u << 26)) ? a0 : n0) + ((bits1 & (1u << 26)) ? a1 : n1);
                crow[j + 27] += ((bits0 & (1u << 27)) ? a0 : n0) + ((bits1 & (1u << 27)) ? a1 : n1);
                crow[j + 28] += ((bits0 & (1u << 28)) ? a0 : n0) + ((bits1 & (1u << 28)) ? a1 : n1);
                crow[j + 29] += ((bits0 & (1u << 29)) ? a0 : n0) + ((bits1 & (1u << 29)) ? a1 : n1);
                crow[j + 30] += ((bits0 & (1u << 30)) ? a0 : n0) + ((bits1 & (1u << 30)) ? a1 : n1);
                crow[j + 31] += ((bits0 & (1u << 31)) ? a0 : n0) + ((bits1 & (1u << 31)) ? a1 : n1);
            }
        }

        if (K & 1) {
            const size_t p = K - 1;
            const float a = arow[p];
            const float na = -a;
            const uint32_t* brow = B + p * K_ints;
            size_t j = 0;
            for (size_t w = 0; w < K_ints; ++w, j += 32) {
                const uint32_t bits = brow[w];
                crow[j + 0]  += (bits & (1u << 0))  ? a : na;
                crow[j + 1]  += (bits & (1u << 1))  ? a : na;
                crow[j + 2]  += (bits & (1u << 2))  ? a : na;
                crow[j + 3]  += (bits & (1u << 3))  ? a : na;
                crow[j + 4]  += (bits & (1u << 4))  ? a : na;
                crow[j + 5]  += (bits & (1u << 5))  ? a : na;
                crow[j + 6]  += (bits & (1u << 6))  ? a : na;
                crow[j + 7]  += (bits & (1u << 7))  ? a : na;
                crow[j + 8]  += (bits & (1u << 8))  ? a : na;
                crow[j + 9]  += (bits & (1u << 9))  ? a : na;
                crow[j + 10] += (bits & (1u << 10)) ? a : na;
                crow[j + 11] += (bits & (1u << 11)) ? a : na;
                crow[j + 12] += (bits & (1u << 12)) ? a : na;
                crow[j + 13] += (bits & (1u << 13)) ? a : na;
                crow[j + 14] += (bits & (1u << 14)) ? a : na;
                crow[j + 15] += (bits & (1u << 15)) ? a : na;
                crow[j + 16] += (bits & (1u << 16)) ? a : na;
                crow[j + 17] += (bits & (1u << 17)) ? a : na;
                crow[j + 18] += (bits & (1u << 18)) ? a : na;
                crow[j + 19] += (bits & (1u << 19)) ? a : na;
                crow[j + 20] += (bits & (1u << 20)) ? a : na;
                crow[j + 21] += (bits & (1u << 21)) ? a : na;
                crow[j + 22] += (bits & (1u << 22)) ? a : na;
                crow[j + 23] += (bits & (1u << 23)) ? a : na;
                crow[j + 24] += (bits & (1u << 24)) ? a : na;
                crow[j + 25] += (bits & (1u << 25)) ? a : na;
                crow[j + 26] += (bits & (1u << 26)) ? a : na;
                crow[j + 27] += (bits & (1u << 27)) ? a : na;
                crow[j + 28] += (bits & (1u << 28)) ? a : na;
                crow[j + 29] += (bits & (1u << 29)) ? a : na;
                crow[j + 30] += (bits & (1u << 30)) ? a : na;
                crow[j + 31] += (bits & (1u << 31)) ? a : na;
            }
        }
    }
}
