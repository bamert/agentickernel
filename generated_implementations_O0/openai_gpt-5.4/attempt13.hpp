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

        for (size_t j = 0; j < K; j += 64) {
            uint32x4_t acci0 = {0u,0u,0u,0u}, acci1 = {0u,0u,0u,0u}, acci2 = {0u,0u,0u,0u}, acci3 = {0u,0u,0u,0u};
            uint32x4_t acci4 = {0u,0u,0u,0u}, acci5 = {0u,0u,0u,0u}, acci6 = {0u,0u,0u,0u}, acci7 = {0u,0u,0u,0u};
            uint32x4_t acci8 = {0u,0u,0u,0u}, acci9 = {0u,0u,0u,0u}, acci10 = {0u,0u,0u,0u}, acci11 = {0u,0u,0u,0u};
            uint32x4_t acci12 = {0u,0u,0u,0u}, acci13 = {0u,0u,0u,0u}, acci14 = {0u,0u,0u,0u}, acci15 = {0u,0u,0u,0u};
            float sumA = 0.0f;
            const size_t w0 = j >> 5;
            const size_t w1 = w0 + 1;
            for (size_t p = 0; p < K; ++p) {
                const float a = arow[p];
                sumA += a;
                const uint32_t bits0 = B[p * K_ints + w0];
                const uint32_t bits1 = B[p * K_ints + w1];
                const uint32_t aval = (uint32_t)a;
                const uint32x4_t va = {aval,aval,aval,aval};
                acci0  += __builtin_convertvector((uint32x4_t){(bits0>>0)&1u,(bits0>>1)&1u,(bits0>>2)&1u,(bits0>>3)&1u}, uint32x4_t) * va;
                acci1  += __builtin_convertvector((uint32x4_t){(bits0>>4)&1u,(bits0>>5)&1u,(bits0>>6)&1u,(bits0>>7)&1u}, uint32x4_t) * va;
                acci2  += __builtin_convertvector((uint32x4_t){(bits0>>8)&1u,(bits0>>9)&1u,(bits0>>10)&1u,(bits0>>11)&1u}, uint32x4_t) * va;
                acci3  += __builtin_convertvector((uint32x4_t){(bits0>>12)&1u,(bits0>>13)&1u,(bits0>>14)&1u,(bits0>>15)&1u}, uint32x4_t) * va;
                acci4  += __builtin_convertvector((uint32x4_t){(bits0>>16)&1u,(bits0>>17)&1u,(bits0>>18)&1u,(bits0>>19)&1u}, uint32x4_t) * va;
                acci5  += __builtin_convertvector((uint32x4_t){(bits0>>20)&1u,(bits0>>21)&1u,(bits0>>22)&1u,(bits0>>23)&1u}, uint32x4_t) * va;
                acci6  += __builtin_convertvector((uint32x4_t){(bits0>>24)&1u,(bits0>>25)&1u,(bits0>>26)&1u,(bits0>>27)&1u}, uint32x4_t) * va;
                acci7  += __builtin_convertvector((uint32x4_t){(bits0>>28)&1u,(bits0>>29)&1u,(bits0>>30)&1u,(bits0>>31)&1u}, uint32x4_t) * va;
                acci8  += __builtin_convertvector((uint32x4_t){(bits1>>0)&1u,(bits1>>1)&1u,(bits1>>2)&1u,(bits1>>3)&1u}, uint32x4_t) * va;
                acci9  += __builtin_convertvector((uint32x4_t){(bits1>>4)&1u,(bits1>>5)&1u,(bits1>>6)&1u,(bits1>>7)&1u}, uint32x4_t) * va;
                acci10 += __builtin_convertvector((uint32x4_t){(bits1>>8)&1u,(bits1>>9)&1u,(bits1>>10)&1u,(bits1>>11)&1u}, uint32x4_t) * va;
                acci11 += __builtin_convertvector((uint32x4_t){(bits1>>12)&1u,(bits1>>13)&1u,(bits1>>14)&1u,(bits1>>15)&1u}, uint32x4_t) * va;
                acci12 += __builtin_convertvector((uint32x4_t){(bits1>>16)&1u,(bits1>>17)&1u,(bits1>>18)&1u,(bits1>>19)&1u}, uint32x4_t) * va;
                acci13 += __builtin_convertvector((uint32x4_t){(bits1>>20)&1u,(bits1>>21)&1u,(bits1>>22)&1u,(bits1>>23)&1u}, uint32x4_t) * va;
                acci14 += __builtin_convertvector((uint32x4_t){(bits1>>24)&1u,(bits1>>25)&1u,(bits1>>26)&1u,(bits1>>27)&1u}, uint32x4_t) * va;
                acci15 += __builtin_convertvector((uint32x4_t){(bits1>>28)&1u,(bits1>>29)&1u,(bits1>>30)&1u,(bits1>>31)&1u}, uint32x4_t) * va;
            }
            const float base = -sumA;
            *(float32x4_t*)(crow + j + 0)  = (float32x4_t){base,base,base,base} + __builtin_convertvector(acci0, float32x4_t) + __builtin_convertvector(acci0, float32x4_t);
            *(float32x4_t*)(crow + j + 4)  = (float32x4_t){base,base,base,base} + __builtin_convertvector(acci1, float32x4_t) + __builtin_convertvector(acci1, float32x4_t);
            *(float32x4_t*)(crow + j + 8)  = (float32x4_t){base,base,base,base} + __builtin_convertvector(acci2, float32x4_t) + __builtin_convertvector(acci2, float32x4_t);
            *(float32x4_t*)(crow + j + 12) = (float32x4_t){base,base,base,base} + __builtin_convertvector(acci3, float32x4_t) + __builtin_convertvector(acci3, float32x4_t);
            *(float32x4_t*)(crow + j + 16) = (float32x4_t){base,base,base,base} + __builtin_convertvector(acci4, float32x4_t) + __builtin_convertvector(acci4, float32x4_t);
            *(float32x4_t*)(crow + j + 20) = (float32x4_t){base,base,base,base} + __builtin_convertvector(acci5, float32x4_t) + __builtin_convertvector(acci5, float32x4_t);
            *(float32x4_t*)(crow + j + 24) = (float32x4_t){base,base,base,base} + __builtin_convertvector(acci6, float32x4_t) + __builtin_convertvector(acci6, float32x4_t);
            *(float32x4_t*)(crow + j + 28) = (float32x4_t){base,base,base,base} + __builtin_convertvector(acci7, float32x4_t) + __builtin_convertvector(acci7, float32x4_t);
            *(float32x4_t*)(crow + j + 32) = (float32x4_t){base,base,base,base} + __builtin_convertvector(acci8, float32x4_t) + __builtin_convertvector(acci8, float32x4_t);
            *(float32x4_t*)(crow + j + 36) = (float32x4_t){base,base,base,base} + __builtin_convertvector(acci9, float32x4_t) + __builtin_convertvector(acci9, float32x4_t);
            *(float32x4_t*)(crow + j + 40) = (float32x4_t){base,base,base,base} + __builtin_convertvector(acci10, float32x4_t) + __builtin_convertvector(acci10, float32x4_t);
            *(float32x4_t*)(crow + j + 44) = (float32x4_t){base,base,base,base} + __builtin_convertvector(acci11, float32x4_t) + __builtin_convertvector(acci11, float32x4_t);
            *(float32x4_t*)(crow + j + 48) = (float32x4_t){base,base,base,base} + __builtin_convertvector(acci12, float32x4_t) + __builtin_convertvector(acci12, float32x4_t);
            *(float32x4_t*)(crow + j + 52) = (float32x4_t){base,base,base,base} + __builtin_convertvector(acci13, float32x4_t) + __builtin_convertvector(acci13, float32x4_t);
            *(float32x4_t*)(crow + j + 56) = (float32x4_t){base,base,base,base} + __builtin_convertvector(acci14, float32x4_t) + __builtin_convertvector(acci14, float32x4_t);
            *(float32x4_t*)(crow + j + 60) = (float32x4_t){base,base,base,base} + __builtin_convertvector(acci15, float32x4_t) + __builtin_convertvector(acci15, float32x4_t);
        }
    }
}
