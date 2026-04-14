#pragma once

typedef unsigned int uint32_t;
typedef decltype(sizeof(0)) size_t;

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;
    for (size_t i = 0; i < M; ++i) {
        const float* arow = A + i * K;
        float* crow = C + i * K;
        for (size_t j = 0; j < K; ++j) {
            float sum = 0.0f;
            const uint32_t* bcol = B + (j >> 5);
            const uint32_t mask = 1u << (j & 31);
            for (size_t p = 0; p < K; ++p) {
                const float a = arow[p];
                const uint32_t bit = bcol[p * K_ints] & mask;
                sum += bit ? a : -a;
            }
            crow[j] = sum;
        }
    }
}
