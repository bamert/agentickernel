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
                crow[j + 0]  += (bits & 0x00000001u) ? a : na;
                crow[j + 1]  += (bits & 0x00000002u) ? a : na;
                crow[j + 2]  += (bits & 0x00000004u) ? a : na;
                crow[j + 3]  += (bits & 0x00000008u) ? a : na;
                crow[j + 4]  += (bits & 0x00000010u) ? a : na;
                crow[j + 5]  += (bits & 0x00000020u) ? a : na;
                crow[j + 6]  += (bits & 0x00000040u) ? a : na;
                crow[j + 7]  += (bits & 0x00000080u) ? a : na;
                crow[j + 8]  += (bits & 0x00000100u) ? a : na;
                crow[j + 9]  += (bits & 0x00000200u) ? a : na;
                crow[j + 10] += (bits & 0x00000400u) ? a : na;
                crow[j + 11] += (bits & 0x00000800u) ? a : na;
                crow[j + 12] += (bits & 0x00001000u) ? a : na;
                crow[j + 13] += (bits & 0x00002000u) ? a : na;
                crow[j + 14] += (bits & 0x00004000u) ? a : na;
                crow[j + 15] += (bits & 0x00008000u) ? a : na;
                crow[j + 16] += (bits & 0x00010000u) ? a : na;
                crow[j + 17] += (bits & 0x00020000u) ? a : na;
                crow[j + 18] += (bits & 0x00040000u) ? a : na;
                crow[j + 19] += (bits & 0x00080000u) ? a : na;
                crow[j + 20] += (bits & 0x00100000u) ? a : na;
                crow[j + 21] += (bits & 0x00200000u) ? a : na;
                crow[j + 22] += (bits & 0x00400000u) ? a : na;
                crow[j + 23] += (bits & 0x00800000u) ? a : na;
                crow[j + 24] += (bits & 0x01000000u) ? a : na;
                crow[j + 25] += (bits & 0x02000000u) ? a : na;
                crow[j + 26] += (bits & 0x04000000u) ? a : na;
                crow[j + 27] += (bits & 0x08000000u) ? a : na;
                crow[j + 28] += (bits & 0x10000000u) ? a : na;
                crow[j + 29] += (bits & 0x20000000u) ? a : na;
                crow[j + 30] += (bits & 0x40000000u) ? a : na;
                crow[j + 31] += (bits & 0x80000000u) ? a : na;
                j += 32;
            }
        }
    }
}
