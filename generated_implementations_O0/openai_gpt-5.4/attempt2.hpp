#pragma once

typedef unsigned int uint32_t;
typedef decltype(sizeof(0)) size_t;

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;
    for (size_t i = 0; i < M; ++i) {
        const float* arow = A + i * K;
        float* crow = C + i * K;

        for (size_t j = 0; j < K; ++j) {
            crow[j] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) {
            const float a = arow[p];
            const float na = -a;
            const uint32_t* brow = B + p * K_ints;
            size_t j = 0;
            for (size_t w = 0; w < K_ints; ++w) {
                uint32_t bits = brow[w];
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
                j += 32;
            }
        }
    }
}
