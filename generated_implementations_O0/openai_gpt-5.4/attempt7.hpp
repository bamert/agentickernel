#pragma once

typedef unsigned int uint32_t;
typedef decltype(sizeof(0)) size_t;

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;
    for (size_t i = 0; i < M; ++i) {
        const float* arow = A + i * K;
        float* crow = C + i * K;
        for (size_t j = 0; j < K; ++j) crow[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            const float a = arow[p];
            const float na = -a;
            const uint32_t* brow = B + p * K_ints;
            size_t j = 0;
            for (size_t w = 0; w + 3 < K_ints; w += 4) {
                uint32_t bits0 = brow[w + 0];
                uint32_t bits1 = brow[w + 1];
                uint32_t bits2 = brow[w + 2];
                uint32_t bits3 = brow[w + 3];
                for (int b = 0; b < 32; ++b) crow[j + b]      += (bits0 & (1u << b)) ? a : na;
                j += 32;
                for (int b = 0; b < 32; ++b) crow[j + b]      += (bits1 & (1u << b)) ? a : na;
                j += 32;
                for (int b = 0; b < 32; ++b) crow[j + b]      += (bits2 & (1u << b)) ? a : na;
                j += 32;
                for (int b = 0; b < 32; ++b) crow[j + b]      += (bits3 & (1u << b)) ? a : na;
                j += 32;
            }
            for (size_t w = (K_ints & ~size_t(3)); w < K_ints; ++w) {
                uint32_t bits = brow[w];
                for (int b = 0; b < 32; ++b) crow[j + b] += (bits & (1u << b)) ? a : na;
                j += 32;
            }
        }
    }
}
