#pragma once

typedef unsigned int uint32_t;
typedef decltype(sizeof(0)) size_t;

typedef float float32x4_t __attribute__((vector_size(16)));
typedef unsigned int uint32x4_t __attribute__((vector_size(16)));

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;
    const uint32x4_t one = {1u,1u,1u,1u};
    const uint32x4_t zero = {0u,0u,0u,0u};
    for (size_t i = 0; i < M; ++i) {
        const float* arow = A + i * K;
        float* crow = C + i * K;
        for (size_t j = 0; j < K; ++j) crow[j] = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            const float a = arow[p];
            const float32x4_t va = {a,a,a,a};
            const float32x4_t vna = {-a,-a,-a,-a};
            const float32x4_t vtwice = {a+a,a+a,a+a,a+a};
            const uint32_t* brow = B + p * K_ints;
            size_t j = 0;
            for (size_t w = 0; w < K_ints; ++w, j += 32) {
                uint32_t bits = brow[w];
                uint32x4_t b0 = {(bits >> 0) & 1u, (bits >> 1) & 1u, (bits >> 2) & 1u, (bits >> 3) & 1u};
                uint32x4_t b1 = {(bits >> 4) & 1u, (bits >> 5) & 1u, (bits >> 6) & 1u, (bits >> 7) & 1u};
                uint32x4_t b2 = {(bits >> 8) & 1u, (bits >> 9) & 1u, (bits >> 10) & 1u, (bits >> 11) & 1u};
                uint32x4_t b3 = {(bits >> 12) & 1u, (bits >> 13) & 1u, (bits >> 14) & 1u, (bits >> 15) & 1u};
                uint32x4_t b4 = {(bits >> 16) & 1u, (bits >> 17) & 1u, (bits >> 18) & 1u, (bits >> 19) & 1u};
                uint32x4_t b5 = {(bits >> 20) & 1u, (bits >> 21) & 1u, (bits >> 22) & 1u, (bits >> 23) & 1u};
                uint32x4_t b6 = {(bits >> 24) & 1u, (bits >> 25) & 1u, (bits >> 26) & 1u, (bits >> 27) & 1u};
                uint32x4_t b7 = {(bits >> 28) & 1u, (bits >> 29) & 1u, (bits >> 30) & 1u, (bits >> 31) & 1u};
                float32x4_t* c0 = (float32x4_t*)(crow + j + 0);
                float32x4_t* c1 = (float32x4_t*)(crow + j + 4);
                float32x4_t* c2 = (float32x4_t*)(crow + j + 8);
                float32x4_t* c3 = (float32x4_t*)(crow + j + 12);
                float32x4_t* c4 = (float32x4_t*)(crow + j + 16);
                float32x4_t* c5 = (float32x4_t*)(crow + j + 20);
                float32x4_t* c6 = (float32x4_t*)(crow + j + 24);
                float32x4_t* c7 = (float32x4_t*)(crow + j + 28);
                *c0 += vna + __builtin_convertvector(b0, float32x4_t) * vtwice;
                *c1 += vna + __builtin_convertvector(b1, float32x4_t) * vtwice;
                *c2 += vna + __builtin_convertvector(b2, float32x4_t) * vtwice;
                *c3 += vna + __builtin_convertvector(b3, float32x4_t) * vtwice;
                *c4 += vna + __builtin_convertvector(b4, float32x4_t) * vtwice;
                *c5 += vna + __builtin_convertvector(b5, float32x4_t) * vtwice;
                *c6 += vna + __builtin_convertvector(b6, float32x4_t) * vtwice;
                *c7 += vna + __builtin_convertvector(b7, float32x4_t) * vtwice;
            }
        }
    }
}
