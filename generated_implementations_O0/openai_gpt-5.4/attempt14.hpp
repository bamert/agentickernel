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

        for (size_t p = 0; p + 1 < K; p += 2) {
            const float a0 = arow[p];
            const float a1 = arow[p + 1];
            const float32x4_t vna = {-(a0 + a1), -(a0 + a1), -(a0 + a1), -(a0 + a1)};
            const float32x4_t vtwice0 = {a0 + a0, a0 + a0, a0 + a0, a0 + a0};
            const float32x4_t vtwice1 = {a1 + a1, a1 + a1, a1 + a1, a1 + a1};
            const uint32_t* brow0 = B + p * K_ints;
            const uint32_t* brow1 = brow0 + K_ints;
            size_t j = 0;
            if (p == 0) {
                for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                    uint32_t x0 = brow0[w + 0], y0 = brow1[w + 0];
                    uint32_t x1 = brow0[w + 1], y1 = brow1[w + 1];
                    *(float32x4_t*)(crow + j + 0)  = vna + __builtin_convertvector((uint32x4_t){(x0>>0)&1u,(x0>>1)&1u,(x0>>2)&1u,(x0>>3)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y0>>0)&1u,(y0>>1)&1u,(y0>>2)&1u,(y0>>3)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 4)  = vna + __builtin_convertvector((uint32x4_t){(x0>>4)&1u,(x0>>5)&1u,(x0>>6)&1u,(x0>>7)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y0>>4)&1u,(y0>>5)&1u,(y0>>6)&1u,(y0>>7)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 8)  = vna + __builtin_convertvector((uint32x4_t){(x0>>8)&1u,(x0>>9)&1u,(x0>>10)&1u,(x0>>11)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y0>>8)&1u,(y0>>9)&1u,(y0>>10)&1u,(y0>>11)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 12) = vna + __builtin_convertvector((uint32x4_t){(x0>>12)&1u,(x0>>13)&1u,(x0>>14)&1u,(x0>>15)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y0>>12)&1u,(y0>>13)&1u,(y0>>14)&1u,(y0>>15)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 16) = vna + __builtin_convertvector((uint32x4_t){(x0>>16)&1u,(x0>>17)&1u,(x0>>18)&1u,(x0>>19)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y0>>16)&1u,(y0>>17)&1u,(y0>>18)&1u,(y0>>19)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 20) = vna + __builtin_convertvector((uint32x4_t){(x0>>20)&1u,(x0>>21)&1u,(x0>>22)&1u,(x0>>23)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y0>>20)&1u,(y0>>21)&1u,(y0>>22)&1u,(y0>>23)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 24) = vna + __builtin_convertvector((uint32x4_t){(x0>>24)&1u,(x0>>25)&1u,(x0>>26)&1u,(x0>>27)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y0>>24)&1u,(y0>>25)&1u,(y0>>26)&1u,(y0>>27)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 28) = vna + __builtin_convertvector((uint32x4_t){(x0>>28)&1u,(x0>>29)&1u,(x0>>30)&1u,(x0>>31)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y0>>28)&1u,(y0>>29)&1u,(y0>>30)&1u,(y0>>31)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 32) = vna + __builtin_convertvector((uint32x4_t){(x1>>0)&1u,(x1>>1)&1u,(x1>>2)&1u,(x1>>3)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y1>>0)&1u,(y1>>1)&1u,(y1>>2)&1u,(y1>>3)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 36) = vna + __builtin_convertvector((uint32x4_t){(x1>>4)&1u,(x1>>5)&1u,(x1>>6)&1u,(x1>>7)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y1>>4)&1u,(y1>>5)&1u,(y1>>6)&1u,(y1>>7)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 40) = vna + __builtin_convertvector((uint32x4_t){(x1>>8)&1u,(x1>>9)&1u,(x1>>10)&1u,(x1>>11)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y1>>8)&1u,(y1>>9)&1u,(y1>>10)&1u,(y1>>11)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 44) = vna + __builtin_convertvector((uint32x4_t){(x1>>12)&1u,(x1>>13)&1u,(x1>>14)&1u,(x1>>15)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y1>>12)&1u,(y1>>13)&1u,(y1>>14)&1u,(y1>>15)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 48) = vna + __builtin_convertvector((uint32x4_t){(x1>>16)&1u,(x1>>17)&1u,(x1>>18)&1u,(x1>>19)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y1>>16)&1u,(y1>>17)&1u,(y1>>18)&1u,(y1>>19)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 52) = vna + __builtin_convertvector((uint32x4_t){(x1>>20)&1u,(x1>>21)&1u,(x1>>22)&1u,(x1>>23)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y1>>20)&1u,(y1>>21)&1u,(y1>>22)&1u,(y1>>23)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 56) = vna + __builtin_convertvector((uint32x4_t){(x1>>24)&1u,(x1>>25)&1u,(x1>>26)&1u,(x1>>27)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y1>>24)&1u,(y1>>25)&1u,(y1>>26)&1u,(y1>>27)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 60) = vna + __builtin_convertvector((uint32x4_t){(x1>>28)&1u,(x1>>29)&1u,(x1>>30)&1u,(x1>>31)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y1>>28)&1u,(y1>>29)&1u,(y1>>30)&1u,(y1>>31)&1u}, float32x4_t) * vtwice1;
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    uint32_t x = brow0[w], y = brow1[w];
                    *(float32x4_t*)(crow + j + 0) = vna + __builtin_convertvector((uint32x4_t){(x>>0)&1u,(x>>1)&1u,(x>>2)&1u,(x>>3)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y>>0)&1u,(y>>1)&1u,(y>>2)&1u,(y>>3)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 4) = vna + __builtin_convertvector((uint32x4_t){(x>>4)&1u,(x>>5)&1u,(x>>6)&1u,(x>>7)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y>>4)&1u,(y>>5)&1u,(y>>6)&1u,(y>>7)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 8) = vna + __builtin_convertvector((uint32x4_t){(x>>8)&1u,(x>>9)&1u,(x>>10)&1u,(x>>11)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y>>8)&1u,(y>>9)&1u,(y>>10)&1u,(y>>11)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 12) = vna + __builtin_convertvector((uint32x4_t){(x>>12)&1u,(x>>13)&1u,(x>>14)&1u,(x>>15)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y>>12)&1u,(y>>13)&1u,(y>>14)&1u,(y>>15)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 16) = vna + __builtin_convertvector((uint32x4_t){(x>>16)&1u,(x>>17)&1u,(x>>18)&1u,(x>>19)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y>>16)&1u,(y>>17)&1u,(y>>18)&1u,(y>>19)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 20) = vna + __builtin_convertvector((uint32x4_t){(x>>20)&1u,(x>>21)&1u,(x>>22)&1u,(x>>23)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y>>20)&1u,(y>>21)&1u,(y>>22)&1u,(y>>23)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 24) = vna + __builtin_convertvector((uint32x4_t){(x>>24)&1u,(x>>25)&1u,(x>>26)&1u,(x>>27)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y>>24)&1u,(y>>25)&1u,(y>>26)&1u,(y>>27)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 28) = vna + __builtin_convertvector((uint32x4_t){(x>>28)&1u,(x>>29)&1u,(x>>30)&1u,(x>>31)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y>>28)&1u,(y>>29)&1u,(y>>30)&1u,(y>>31)&1u}, float32x4_t) * vtwice1;
                }
            } else {
                for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                    uint32_t x0 = brow0[w + 0], y0 = brow1[w + 0];
                    uint32_t x1 = brow0[w + 1], y1 = brow1[w + 1];
                    *(float32x4_t*)(crow + j + 0)  += vna + __builtin_convertvector((uint32x4_t){(x0>>0)&1u,(x0>>1)&1u,(x0>>2)&1u,(x0>>3)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y0>>0)&1u,(y0>>1)&1u,(y0>>2)&1u,(y0>>3)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 4)  += vna + __builtin_convertvector((uint32x4_t){(x0>>4)&1u,(x0>>5)&1u,(x0>>6)&1u,(x0>>7)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y0>>4)&1u,(y0>>5)&1u,(y0>>6)&1u,(y0>>7)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 8)  += vna + __builtin_convertvector((uint32x4_t){(x0>>8)&1u,(x0>>9)&1u,(x0>>10)&1u,(x0>>11)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y0>>8)&1u,(y0>>9)&1u,(y0>>10)&1u,(y0>>11)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 12) += vna + __builtin_convertvector((uint32x4_t){(x0>>12)&1u,(x0>>13)&1u,(x0>>14)&1u,(x0>>15)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y0>>12)&1u,(y0>>13)&1u,(y0>>14)&1u,(y0>>15)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 16) += vna + __builtin_convertvector((uint32x4_t){(x0>>16)&1u,(x0>>17)&1u,(x0>>18)&1u,(x0>>19)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y0>>16)&1u,(y0>>17)&1u,(y0>>18)&1u,(y0>>19)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 20) += vna + __builtin_convertvector((uint32x4_t){(x0>>20)&1u,(x0>>21)&1u,(x0>>22)&1u,(x0>>23)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y0>>20)&1u,(y0>>21)&1u,(y0>>22)&1u,(y0>>23)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 24) += vna + __builtin_convertvector((uint32x4_t){(x0>>24)&1u,(x0>>25)&1u,(x0>>26)&1u,(x0>>27)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y0>>24)&1u,(y0>>25)&1u,(y0>>26)&1u,(y0>>27)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 28) += vna + __builtin_convertvector((uint32x4_t){(x0>>28)&1u,(x0>>29)&1u,(x0>>30)&1u,(x0>>31)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y0>>28)&1u,(y0>>29)&1u,(y0>>30)&1u,(y0>>31)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 32) += vna + __builtin_convertvector((uint32x4_t){(x1>>0)&1u,(x1>>1)&1u,(x1>>2)&1u,(x1>>3)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y1>>0)&1u,(y1>>1)&1u,(y1>>2)&1u,(y1>>3)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 36) += vna + __builtin_convertvector((uint32x4_t){(x1>>4)&1u,(x1>>5)&1u,(x1>>6)&1u,(x1>>7)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y1>>4)&1u,(y1>>5)&1u,(y1>>6)&1u,(y1>>7)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 40) += vna + __builtin_convertvector((uint32x4_t){(x1>>8)&1u,(x1>>9)&1u,(x1>>10)&1u,(x1>>11)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y1>>8)&1u,(y1>>9)&1u,(y1>>10)&1u,(y1>>11)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 44) += vna + __builtin_convertvector((uint32x4_t){(x1>>12)&1u,(x1>>13)&1u,(x1>>14)&1u,(x1>>15)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y1>>12)&1u,(y1>>13)&1u,(y1>>14)&1u,(y1>>15)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 48) += vna + __builtin_convertvector((uint32x4_t){(x1>>16)&1u,(x1>>17)&1u,(x1>>18)&1u,(x1>>19)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y1>>16)&1u,(y1>>17)&1u,(y1>>18)&1u,(y1>>19)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 52) += vna + __builtin_convertvector((uint32x4_t){(x1>>20)&1u,(x1>>21)&1u,(x1>>22)&1u,(x1>>23)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y1>>20)&1u,(y1>>21)&1u,(y1>>22)&1u,(y1>>23)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 56) += vna + __builtin_convertvector((uint32x4_t){(x1>>24)&1u,(x1>>25)&1u,(x1>>26)&1u,(x1>>27)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y1>>24)&1u,(y1>>25)&1u,(y1>>26)&1u,(y1>>27)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 60) += vna + __builtin_convertvector((uint32x4_t){(x1>>28)&1u,(x1>>29)&1u,(x1>>30)&1u,(x1>>31)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y1>>28)&1u,(y1>>29)&1u,(y1>>30)&1u,(y1>>31)&1u}, float32x4_t) * vtwice1;
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    uint32_t x = brow0[w], y = brow1[w];
                    *(float32x4_t*)(crow + j + 0) += vna + __builtin_convertvector((uint32x4_t){(x>>0)&1u,(x>>1)&1u,(x>>2)&1u,(x>>3)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y>>0)&1u,(y>>1)&1u,(y>>2)&1u,(y>>3)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 4) += vna + __builtin_convertvector((uint32x4_t){(x>>4)&1u,(x>>5)&1u,(x>>6)&1u,(x>>7)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y>>4)&1u,(y>>5)&1u,(y>>6)&1u,(y>>7)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 8) += vna + __builtin_convertvector((uint32x4_t){(x>>8)&1u,(x>>9)&1u,(x>>10)&1u,(x>>11)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y>>8)&1u,(y>>9)&1u,(y>>10)&1u,(y>>11)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 12) += vna + __builtin_convertvector((uint32x4_t){(x>>12)&1u,(x>>13)&1u,(x>>14)&1u,(x>>15)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y>>12)&1u,(y>>13)&1u,(y>>14)&1u,(y>>15)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 16) += vna + __builtin_convertvector((uint32x4_t){(x>>16)&1u,(x>>17)&1u,(x>>18)&1u,(x>>19)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y>>16)&1u,(y>>17)&1u,(y>>18)&1u,(y>>19)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 20) += vna + __builtin_convertvector((uint32x4_t){(x>>20)&1u,(x>>21)&1u,(x>>22)&1u,(x>>23)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y>>20)&1u,(y>>21)&1u,(y>>22)&1u,(y>>23)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 24) += vna + __builtin_convertvector((uint32x4_t){(x>>24)&1u,(x>>25)&1u,(x>>26)&1u,(x>>27)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y>>24)&1u,(y>>25)&1u,(y>>26)&1u,(y>>27)&1u}, float32x4_t) * vtwice1;
                    *(float32x4_t*)(crow + j + 28) += vna + __builtin_convertvector((uint32x4_t){(x>>28)&1u,(x>>29)&1u,(x>>30)&1u,(x>>31)&1u}, float32x4_t) * vtwice0 + __builtin_convertvector((uint32x4_t){(y>>28)&1u,(y>>29)&1u,(y>>30)&1u,(y>>31)&1u}, float32x4_t) * vtwice1;
                }
            }
        }

        if (K & 1) {
            const size_t p = K - 1;
            const float a = arow[p];
            const float32x4_t vna = {-a,-a,-a,-a};
            const float32x4_t vtwice = {a+a,a+a,a+a,a+a};
            const uint32_t* brow = B + p * K_ints;
            size_t j = 0;
            if (p == 0) {
                for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                    uint32_t bits0 = brow[w + 0], bits1 = brow[w + 1];
                    *(float32x4_t*)(crow + j + 0)  = vna + __builtin_convertvector((uint32x4_t){(bits0>>0)&1u,(bits0>>1)&1u,(bits0>>2)&1u,(bits0>>3)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 4)  = vna + __builtin_convertvector((uint32x4_t){(bits0>>4)&1u,(bits0>>5)&1u,(bits0>>6)&1u,(bits0>>7)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 8)  = vna + __builtin_convertvector((uint32x4_t){(bits0>>8)&1u,(bits0>>9)&1u,(bits0>>10)&1u,(bits0>>11)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 12) = vna + __builtin_convertvector((uint32x4_t){(bits0>>12)&1u,(bits0>>13)&1u,(bits0>>14)&1u,(bits0>>15)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 16) = vna + __builtin_convertvector((uint32x4_t){(bits0>>16)&1u,(bits0>>17)&1u,(bits0>>18)&1u,(bits0>>19)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 20) = vna + __builtin_convertvector((uint32x4_t){(bits0>>20)&1u,(bits0>>21)&1u,(bits0>>22)&1u,(bits0>>23)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 24) = vna + __builtin_convertvector((uint32x4_t){(bits0>>24)&1u,(bits0>>25)&1u,(bits0>>26)&1u,(bits0>>27)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 28) = vna + __builtin_convertvector((uint32x4_t){(bits0>>28)&1u,(bits0>>29)&1u,(bits0>>30)&1u,(bits0>>31)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 32) = vna + __builtin_convertvector((uint32x4_t){(bits1>>0)&1u,(bits1>>1)&1u,(bits1>>2)&1u,(bits1>>3)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 36) = vna + __builtin_convertvector((uint32x4_t){(bits1>>4)&1u,(bits1>>5)&1u,(bits1>>6)&1u,(bits1>>7)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 40) = vna + __builtin_convertvector((uint32x4_t){(bits1>>8)&1u,(bits1>>9)&1u,(bits1>>10)&1u,(bits1>>11)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 44) = vna + __builtin_convertvector((uint32x4_t){(bits1>>12)&1u,(bits1>>13)&1u,(bits1>>14)&1u,(bits1>>15)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 48) = vna + __builtin_convertvector((uint32x4_t){(bits1>>16)&1u,(bits1>>17)&1u,(bits1>>18)&1u,(bits1>>19)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 52) = vna + __builtin_convertvector((uint32x4_t){(bits1>>20)&1u,(bits1>>21)&1u,(bits1>>22)&1u,(bits1>>23)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 56) = vna + __builtin_convertvector((uint32x4_t){(bits1>>24)&1u,(bits1>>25)&1u,(bits1>>26)&1u,(bits1>>27)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 60) = vna + __builtin_convertvector((uint32x4_t){(bits1>>28)&1u,(bits1>>29)&1u,(bits1>>30)&1u,(bits1>>31)&1u}, float32x4_t) * vtwice;
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    uint32_t bits = brow[w];
                    *(float32x4_t*)(crow + j + 0) = vna + __builtin_convertvector((uint32x4_t){(bits>>0)&1u,(bits>>1)&1u,(bits>>2)&1u,(bits>>3)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 4) = vna + __builtin_convertvector((uint32x4_t){(bits>>4)&1u,(bits>>5)&1u,(bits>>6)&1u,(bits>>7)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 8) = vna + __builtin_convertvector((uint32x4_t){(bits>>8)&1u,(bits>>9)&1u,(bits>>10)&1u,(bits>>11)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 12) = vna + __builtin_convertvector((uint32x4_t){(bits>>12)&1u,(bits>>13)&1u,(bits>>14)&1u,(bits>>15)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 16) = vna + __builtin_convertvector((uint32x4_t){(bits>>16)&1u,(bits>>17)&1u,(bits>>18)&1u,(bits>>19)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 20) = vna + __builtin_convertvector((uint32x4_t){(bits>>20)&1u,(bits>>21)&1u,(bits>>22)&1u,(bits>>23)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 24) = vna + __builtin_convertvector((uint32x4_t){(bits>>24)&1u,(bits>>25)&1u,(bits>>26)&1u,(bits>>27)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 28) = vna + __builtin_convertvector((uint32x4_t){(bits>>28)&1u,(bits>>29)&1u,(bits>>30)&1u,(bits>>31)&1u}, float32x4_t) * vtwice;
                }
            } else {
                for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                    uint32_t bits0 = brow[w + 0], bits1 = brow[w + 1];
                    *(float32x4_t*)(crow + j + 0)  += vna + __builtin_convertvector((uint32x4_t){(bits0>>0)&1u,(bits0>>1)&1u,(bits0>>2)&1u,(bits0>>3)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 4)  += vna + __builtin_convertvector((uint32x4_t){(bits0>>4)&1u,(bits0>>5)&1u,(bits0>>6)&1u,(bits0>>7)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 8)  += vna + __builtin_convertvector((uint32x4_t){(bits0>>8)&1u,(bits0>>9)&1u,(bits0>>10)&1u,(bits0>>11)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 12) += vna + __builtin_convertvector((uint32x4_t){(bits0>>12)&1u,(bits0>>13)&1u,(bits0>>14)&1u,(bits0>>15)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 16) += vna + __builtin_convertvector((uint32x4_t){(bits0>>16)&1u,(bits0>>17)&1u,(bits0>>18)&1u,(bits0>>19)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 20) += vna + __builtin_convertvector((uint32x4_t){(bits0>>20)&1u,(bits0>>21)&1u,(bits0>>22)&1u,(bits0>>23)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 24) += vna + __builtin_convertvector((uint32x4_t){(bits0>>24)&1u,(bits0>>25)&1u,(bits0>>26)&1u,(bits0>>27)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 28) += vna + __builtin_convertvector((uint32x4_t){(bits0>>28)&1u,(bits0>>29)&1u,(bits0>>30)&1u,(bits0>>31)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 32) += vna + __builtin_convertvector((uint32x4_t){(bits1>>0)&1u,(bits1>>1)&1u,(bits1>>2)&1u,(bits1>>3)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 36) += vna + __builtin_convertvector((uint32x4_t){(bits1>>4)&1u,(bits1>>5)&1u,(bits1>>6)&1u,(bits1>>7)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 40) += vna + __builtin_convertvector((uint32x4_t){(bits1>>8)&1u,(bits1>>9)&1u,(bits1>>10)&1u,(bits1>>11)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 44) += vna + __builtin_convertvector((uint32x4_t){(bits1>>12)&1u,(bits1>>13)&1u,(bits1>>14)&1u,(bits1>>15)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 48) += vna + __builtin_convertvector((uint32x4_t){(bits1>>16)&1u,(bits1>>17)&1u,(bits1>>18)&1u,(bits1>>19)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 52) += vna + __builtin_convertvector((uint32x4_t){(bits1>>20)&1u,(bits1>>21)&1u,(bits1>>22)&1u,(bits1>>23)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 56) += vna + __builtin_convertvector((uint32x4_t){(bits1>>24)&1u,(bits1>>25)&1u,(bits1>>26)&1u,(bits1>>27)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 60) += vna + __builtin_convertvector((uint32x4_t){(bits1>>28)&1u,(bits1>>29)&1u,(bits1>>30)&1u,(bits1>>31)&1u}, float32x4_t) * vtwice;
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    uint32_t bits = brow[w];
                    *(float32x4_t*)(crow + j + 0) += vna + __builtin_convertvector((uint32x4_t){(bits>>0)&1u,(bits>>1)&1u,(bits>>2)&1u,(bits>>3)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 4) += vna + __builtin_convertvector((uint32x4_t){(bits>>4)&1u,(bits>>5)&1u,(bits>>6)&1u,(bits>>7)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 8) += vna + __builtin_convertvector((uint32x4_t){(bits>>8)&1u,(bits>>9)&1u,(bits>>10)&1u,(bits>>11)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 12) += vna + __builtin_convertvector((uint32x4_t){(bits>>12)&1u,(bits>>13)&1u,(bits>>14)&1u,(bits>>15)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 16) += vna + __builtin_convertvector((uint32x4_t){(bits>>16)&1u,(bits>>17)&1u,(bits>>18)&1u,(bits>>19)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 20) += vna + __builtin_convertvector((uint32x4_t){(bits>>20)&1u,(bits>>21)&1u,(bits>>22)&1u,(bits>>23)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 24) += vna + __builtin_convertvector((uint32x4_t){(bits>>24)&1u,(bits>>25)&1u,(bits>>26)&1u,(bits>>27)&1u}, float32x4_t) * vtwice;
                    *(float32x4_t*)(crow + j + 28) += vna + __builtin_convertvector((uint32x4_t){(bits>>28)&1u,(bits>>29)&1u,(bits>>30)&1u,(bits>>31)&1u}, float32x4_t) * vtwice;
                }
            }
        }
    }
}
