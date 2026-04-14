#pragma once

typedef unsigned int uint32_t;
typedef decltype(sizeof(0)) size_t;

typedef float float32x4_t __attribute__((vector_size(16)));
typedef unsigned int uint32x4_t __attribute__((vector_size(16)));

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;
    for (size_t i = 0; i < M; ++i) {
        const float* arow = A + i * K;
        float* crow = C + i * K;
        for (size_t j = 0; j < K; ++j) crow[j] = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            const float a = arow[p];
            const float32x4_t vna = {-a,-a,-a,-a};
            const float32x4_t vtwice = {a+a,a+a,a+a,a+a};
            const uint32_t* brow = B + p * K_ints;
            size_t j = 0;
            for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                uint32_t bits0 = brow[w + 0];
                uint32_t bits1 = brow[w + 1];
                float32x4_t* c0 = (float32x4_t*)(crow + j + 0);
                float32x4_t* c1 = (float32x4_t*)(crow + j + 4);
                float32x4_t* c2 = (float32x4_t*)(crow + j + 8);
                float32x4_t* c3 = (float32x4_t*)(crow + j + 12);
                float32x4_t* c4 = (float32x4_t*)(crow + j + 16);
                float32x4_t* c5 = (float32x4_t*)(crow + j + 20);
                float32x4_t* c6 = (float32x4_t*)(crow + j + 24);
                float32x4_t* c7 = (float32x4_t*)(crow + j + 28);
                float32x4_t* c8 = (float32x4_t*)(crow + j + 32);
                float32x4_t* c9 = (float32x4_t*)(crow + j + 36);
                float32x4_t* c10 = (float32x4_t*)(crow + j + 40);
                float32x4_t* c11 = (float32x4_t*)(crow + j + 44);
                float32x4_t* c12 = (float32x4_t*)(crow + j + 48);
                float32x4_t* c13 = (float32x4_t*)(crow + j + 52);
                float32x4_t* c14 = (float32x4_t*)(crow + j + 56);
                float32x4_t* c15 = (float32x4_t*)(crow + j + 60);
                *c0  += vna + __builtin_convertvector((uint32x4_t){(bits0>>0)&1u,(bits0>>1)&1u,(bits0>>2)&1u,(bits0>>3)&1u}, float32x4_t) * vtwice;
                *c1  += vna + __builtin_convertvector((uint32x4_t){(bits0>>4)&1u,(bits0>>5)&1u,(bits0>>6)&1u,(bits0>>7)&1u}, float32x4_t) * vtwice;
                *c2  += vna + __builtin_convertvector((uint32x4_t){(bits0>>8)&1u,(bits0>>9)&1u,(bits0>>10)&1u,(bits0>>11)&1u}, float32x4_t) * vtwice;
                *c3  += vna + __builtin_convertvector((uint32x4_t){(bits0>>12)&1u,(bits0>>13)&1u,(bits0>>14)&1u,(bits0>>15)&1u}, float32x4_t) * vtwice;
                *c4  += vna + __builtin_convertvector((uint32x4_t){(bits0>>16)&1u,(bits0>>17)&1u,(bits0>>18)&1u,(bits0>>19)&1u}, float32x4_t) * vtwice;
                *c5  += vna + __builtin_convertvector((uint32x4_t){(bits0>>20)&1u,(bits0>>21)&1u,(bits0>>22)&1u,(bits0>>23)&1u}, float32x4_t) * vtwice;
                *c6  += vna + __builtin_convertvector((uint32x4_t){(bits0>>24)&1u,(bits0>>25)&1u,(bits0>>26)&1u,(bits0>>27)&1u}, float32x4_t) * vtwice;
                *c7  += vna + __builtin_convertvector((uint32x4_t){(bits0>>28)&1u,(bits0>>29)&1u,(bits0>>30)&1u,(bits0>>31)&1u}, float32x4_t) * vtwice;
                *c8  += vna + __builtin_convertvector((uint32x4_t){(bits1>>0)&1u,(bits1>>1)&1u,(bits1>>2)&1u,(bits1>>3)&1u}, float32x4_t) * vtwice;
                *c9  += vna + __builtin_convertvector((uint32x4_t){(bits1>>4)&1u,(bits1>>5)&1u,(bits1>>6)&1u,(bits1>>7)&1u}, float32x4_t) * vtwice;
                *c10 += vna + __builtin_convertvector((uint32x4_t){(bits1>>8)&1u,(bits1>>9)&1u,(bits1>>10)&1u,(bits1>>11)&1u}, float32x4_t) * vtwice;
                *c11 += vna + __builtin_convertvector((uint32x4_t){(bits1>>12)&1u,(bits1>>13)&1u,(bits1>>14)&1u,(bits1>>15)&1u}, float32x4_t) * vtwice;
                *c12 += vna + __builtin_convertvector((uint32x4_t){(bits1>>16)&1u,(bits1>>17)&1u,(bits1>>18)&1u,(bits1>>19)&1u}, float32x4_t) * vtwice;
                *c13 += vna + __builtin_convertvector((uint32x4_t){(bits1>>20)&1u,(bits1>>21)&1u,(bits1>>22)&1u,(bits1>>23)&1u}, float32x4_t) * vtwice;
                *c14 += vna + __builtin_convertvector((uint32x4_t){(bits1>>24)&1u,(bits1>>25)&1u,(bits1>>26)&1u,(bits1>>27)&1u}, float32x4_t) * vtwice;
                *c15 += vna + __builtin_convertvector((uint32x4_t){(bits1>>28)&1u,(bits1>>29)&1u,(bits1>>30)&1u,(bits1>>31)&1u}, float32x4_t) * vtwice;
            }
            for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                uint32_t bits = brow[w];
                float32x4_t* c0 = (float32x4_t*)(crow + j + 0);
                float32x4_t* c1 = (float32x4_t*)(crow + j + 4);
                float32x4_t* c2 = (float32x4_t*)(crow + j + 8);
                float32x4_t* c3 = (float32x4_t*)(crow + j + 12);
                float32x4_t* c4 = (float32x4_t*)(crow + j + 16);
                float32x4_t* c5 = (float32x4_t*)(crow + j + 20);
                float32x4_t* c6 = (float32x4_t*)(crow + j + 24);
                float32x4_t* c7 = (float32x4_t*)(crow + j + 28);
                *c0 += vna + __builtin_convertvector((uint32x4_t){(bits>>0)&1u,(bits>>1)&1u,(bits>>2)&1u,(bits>>3)&1u}, float32x4_t) * vtwice;
                *c1 += vna + __builtin_convertvector((uint32x4_t){(bits>>4)&1u,(bits>>5)&1u,(bits>>6)&1u,(bits>>7)&1u}, float32x4_t) * vtwice;
                *c2 += vna + __builtin_convertvector((uint32x4_t){(bits>>8)&1u,(bits>>9)&1u,(bits>>10)&1u,(bits>>11)&1u}, float32x4_t) * vtwice;
                *c3 += vna + __builtin_convertvector((uint32x4_t){(bits>>12)&1u,(bits>>13)&1u,(bits>>14)&1u,(bits>>15)&1u}, float32x4_t) * vtwice;
                *c4 += vna + __builtin_convertvector((uint32x4_t){(bits>>16)&1u,(bits>>17)&1u,(bits>>18)&1u,(bits>>19)&1u}, float32x4_t) * vtwice;
                *c5 += vna + __builtin_convertvector((uint32x4_t){(bits>>20)&1u,(bits>>21)&1u,(bits>>22)&1u,(bits>>23)&1u}, float32x4_t) * vtwice;
                *c6 += vna + __builtin_convertvector((uint32x4_t){(bits>>24)&1u,(bits>>25)&1u,(bits>>26)&1u,(bits>>27)&1u}, float32x4_t) * vtwice;
                *c7 += vna + __builtin_convertvector((uint32x4_t){(bits>>28)&1u,(bits>>29)&1u,(bits>>30)&1u,(bits>>31)&1u}, float32x4_t) * vtwice;
            }
        }
    }
}
