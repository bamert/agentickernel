#pragma once

typedef unsigned int uint32_t;
typedef decltype(sizeof(0)) size_t;

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K >> 5;
    for (size_t i = 0; i < M; ++i) {
        const float* arow = A + i * K;
        float* crow = C + i * K;
        for (size_t j = 0; j < K; ++j) crow[j] = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            float a = arow[p];
            const uint32_t* brow = B + p * K_ints;
            size_t j = 0;
            for (size_t w = 0; w < K_ints; ++w) {
                uint32_t bits = brow[w];
                for (size_t b = 0; b < 32; ++b, ++j) {
                    crow[j] += ((bits & (1u << b)) ? a : -a);
                }
            }
        }
    }
}
