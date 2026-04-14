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

        for (size_t p = 0; p + 3 < K; p += 4) {
            const float a0 = arow[p + 0], a1 = arow[p + 1], a2 = arow[p + 2], a3 = arow[p + 3];
            const float32x4_t vbase = {-(a0+a1+a2+a3),-(a0+a1+a2+a3),-(a0+a1+a2+a3),-(a0+a1+a2+a3)};
            const float32x4_t v20 = {a0+a0,a0+a0,a0+a0,a0+a0};
            const float32x4_t v21 = {a1+a1,a1+a1,a1+a1,a1+a1};
            const float32x4_t v22 = {a2+a2,a2+a2,a2+a2,a2+a2};
            const float32x4_t v23 = {a3+a3,a3+a3,a3+a3,a3+a3};
            const uint32_t* b0 = B + (p + 0) * K_ints;
            const uint32_t* b1 = b0 + K_ints;
            const uint32_t* b2 = b1 + K_ints;
            const uint32_t* b3 = b2 + K_ints;
            size_t j = 0;
            if (p == 0) {
                for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                    uint32_t x0=b0[w], y0=b1[w], z0=b2[w], t0=b3[w];
                    uint32_t x1=b0[w+1], y1=b1[w+1], z1=b2[w+1], t1=b3[w+1];
#define DOSET(dst, X,Y,Z,T, sh) *(float32x4_t*)(crow + dst) = vbase + __builtin_convertvector((uint32x4_t){((X)>>((sh)+0))&1u,((X)>>((sh)+1))&1u,((X)>>((sh)+2))&1u,((X)>>((sh)+3))&1u}, float32x4_t)*v20 + __builtin_convertvector((uint32x4_t){((Y)>>((sh)+0))&1u,((Y)>>((sh)+1))&1u,((Y)>>((sh)+2))&1u,((Y)>>((sh)+3))&1u}, float32x4_t)*v21 + __builtin_convertvector((uint32x4_t){((Z)>>((sh)+0))&1u,((Z)>>((sh)+1))&1u,((Z)>>((sh)+2))&1u,((Z)>>((sh)+3))&1u}, float32x4_t)*v22 + __builtin_convertvector((uint32x4_t){((T)>>((sh)+0))&1u,((T)>>((sh)+1))&1u,((T)>>((sh)+2))&1u,((T)>>((sh)+3))&1u}, float32x4_t)*v23;
                    DOSET(j+0,x0,y0,z0,t0,0) DOSET(j+4,x0,y0,z0,t0,4) DOSET(j+8,x0,y0,z0,t0,8) DOSET(j+12,x0,y0,z0,t0,12)
                    DOSET(j+16,x0,y0,z0,t0,16) DOSET(j+20,x0,y0,z0,t0,20) DOSET(j+24,x0,y0,z0,t0,24) DOSET(j+28,x0,y0,z0,t0,28)
                    DOSET(j+32,x1,y1,z1,t1,0) DOSET(j+36,x1,y1,z1,t1,4) DOSET(j+40,x1,y1,z1,t1,8) DOSET(j+44,x1,y1,z1,t1,12)
                    DOSET(j+48,x1,y1,z1,t1,16) DOSET(j+52,x1,y1,z1,t1,20) DOSET(j+56,x1,y1,z1,t1,24) DOSET(j+60,x1,y1,z1,t1,28)
#undef DOSET
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    uint32_t x=b0[w], y=b1[w], z=b2[w], t=b3[w];
#define DOSET(dst, X,Y,Z,T, sh) *(float32x4_t*)(crow + dst) = vbase + __builtin_convertvector((uint32x4_t){((X)>>((sh)+0))&1u,((X)>>((sh)+1))&1u,((X)>>((sh)+2))&1u,((X)>>((sh)+3))&1u}, float32x4_t)*v20 + __builtin_convertvector((uint32x4_t){((Y)>>((sh)+0))&1u,((Y)>>((sh)+1))&1u,((Y)>>((sh)+2))&1u,((Y)>>((sh)+3))&1u}, float32x4_t)*v21 + __builtin_convertvector((uint32x4_t){((Z)>>((sh)+0))&1u,((Z)>>((sh)+1))&1u,((Z)>>((sh)+2))&1u,((Z)>>((sh)+3))&1u}, float32x4_t)*v22 + __builtin_convertvector((uint32x4_t){((T)>>((sh)+0))&1u,((T)>>((sh)+1))&1u,((T)>>((sh)+2))&1u,((T)>>((sh)+3))&1u}, float32x4_t)*v23;
                    DOSET(j+0,x,y,z,t,0) DOSET(j+4,x,y,z,t,4) DOSET(j+8,x,y,z,t,8) DOSET(j+12,x,y,z,t,12)
                    DOSET(j+16,x,y,z,t,16) DOSET(j+20,x,y,z,t,20) DOSET(j+24,x,y,z,t,24) DOSET(j+28,x,y,z,t,28)
#undef DOSET
                }
            } else {
                for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                    uint32_t x0=b0[w], y0=b1[w], z0=b2[w], t0=b3[w];
                    uint32_t x1=b0[w+1], y1=b1[w+1], z1=b2[w+1], t1=b3[w+1];
#define DOADD(dst, X,Y,Z,T, sh) *(float32x4_t*)(crow + dst) += vbase + __builtin_convertvector((uint32x4_t){((X)>>((sh)+0))&1u,((X)>>((sh)+1))&1u,((X)>>((sh)+2))&1u,((X)>>((sh)+3))&1u}, float32x4_t)*v20 + __builtin_convertvector((uint32x4_t){((Y)>>((sh)+0))&1u,((Y)>>((sh)+1))&1u,((Y)>>((sh)+2))&1u,((Y)>>((sh)+3))&1u}, float32x4_t)*v21 + __builtin_convertvector((uint32x4_t){((Z)>>((sh)+0))&1u,((Z)>>((sh)+1))&1u,((Z)>>((sh)+2))&1u,((Z)>>((sh)+3))&1u}, float32x4_t)*v22 + __builtin_convertvector((uint32x4_t){((T)>>((sh)+0))&1u,((T)>>((sh)+1))&1u,((T)>>((sh)+2))&1u,((T)>>((sh)+3))&1u}, float32x4_t)*v23;
                    DOADD(j+0,x0,y0,z0,t0,0) DOADD(j+4,x0,y0,z0,t0,4) DOADD(j+8,x0,y0,z0,t0,8) DOADD(j+12,x0,y0,z0,t0,12)
                    DOADD(j+16,x0,y0,z0,t0,16) DOADD(j+20,x0,y0,z0,t0,20) DOADD(j+24,x0,y0,z0,t0,24) DOADD(j+28,x0,y0,z0,t0,28)
                    DOADD(j+32,x1,y1,z1,t1,0) DOADD(j+36,x1,y1,z1,t1,4) DOADD(j+40,x1,y1,z1,t1,8) DOADD(j+44,x1,y1,z1,t1,12)
                    DOADD(j+48,x1,y1,z1,t1,16) DOADD(j+52,x1,y1,z1,t1,20) DOADD(j+56,x1,y1,z1,t1,24) DOADD(j+60,x1,y1,z1,t1,28)
#undef DOADD
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    uint32_t x=b0[w], y=b1[w], z=b2[w], t=b3[w];
#define DOADD(dst, X,Y,Z,T, sh) *(float32x4_t*)(crow + dst) += vbase + __builtin_convertvector((uint32x4_t){((X)>>((sh)+0))&1u,((X)>>((sh)+1))&1u,((X)>>((sh)+2))&1u,((X)>>((sh)+3))&1u}, float32x4_t)*v20 + __builtin_convertvector((uint32x4_t){((Y)>>((sh)+0))&1u,((Y)>>((sh)+1))&1u,((Y)>>((sh)+2))&1u,((Y)>>((sh)+3))&1u}, float32x4_t)*v21 + __builtin_convertvector((uint32x4_t){((Z)>>((sh)+0))&1u,((Z)>>((sh)+1))&1u,((Z)>>((sh)+2))&1u,((Z)>>((sh)+3))&1u}, float32x4_t)*v22 + __builtin_convertvector((uint32x4_t){((T)>>((sh)+0))&1u,((T)>>((sh)+1))&1u,((T)>>((sh)+2))&1u,((T)>>((sh)+3))&1u}, float32x4_t)*v23;
                    DOADD(j+0,x,y,z,t,0) DOADD(j+4,x,y,z,t,4) DOADD(j+8,x,y,z,t,8) DOADD(j+12,x,y,z,t,12)
                    DOADD(j+16,x,y,z,t,16) DOADD(j+20,x,y,z,t,20) DOADD(j+24,x,y,z,t,24) DOADD(j+28,x,y,z,t,28)
#undef DOADD
                }
            }
        }

        for (size_t p = K & ~size_t(3); p < K; ++p) {
            const float a = arow[p];
            const float32x4_t vna = {-a,-a,-a,-a};
            const float32x4_t vtwice = {a+a,a+a,a+a,a+a};
            const uint32_t* brow = B + p * K_ints;
            size_t j = 0;
            if (p == 0) {
                for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                    uint32_t bits0 = brow[w + 0], bits1 = brow[w + 1];
#define DOSET1(dst, X, sh) *(float32x4_t*)(crow + dst) = vna + __builtin_convertvector((uint32x4_t){((X)>>((sh)+0))&1u,((X)>>((sh)+1))&1u,((X)>>((sh)+2))&1u,((X)>>((sh)+3))&1u}, float32x4_t)*vtwice;
                    DOSET1(j+0,bits0,0) DOSET1(j+4,bits0,4) DOSET1(j+8,bits0,8) DOSET1(j+12,bits0,12)
                    DOSET1(j+16,bits0,16) DOSET1(j+20,bits0,20) DOSET1(j+24,bits0,24) DOSET1(j+28,bits0,28)
                    DOSET1(j+32,bits1,0) DOSET1(j+36,bits1,4) DOSET1(j+40,bits1,8) DOSET1(j+44,bits1,12)
                    DOSET1(j+48,bits1,16) DOSET1(j+52,bits1,20) DOSET1(j+56,bits1,24) DOSET1(j+60,bits1,28)
#undef DOSET1
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    uint32_t bits = brow[w];
#define DOSET1(dst, X, sh) *(float32x4_t*)(crow + dst) = vna + __builtin_convertvector((uint32x4_t){((X)>>((sh)+0))&1u,((X)>>((sh)+1))&1u,((X)>>((sh)+2))&1u,((X)>>((sh)+3))&1u}, float32x4_t)*vtwice;
                    DOSET1(j+0,bits,0) DOSET1(j+4,bits,4) DOSET1(j+8,bits,8) DOSET1(j+12,bits,12)
                    DOSET1(j+16,bits,16) DOSET1(j+20,bits,20) DOSET1(j+24,bits,24) DOSET1(j+28,bits,28)
#undef DOSET1
                }
            } else {
                for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                    uint32_t bits0 = brow[w + 0], bits1 = brow[w + 1];
#define DOADD1(dst, X, sh) *(float32x4_t*)(crow + dst) += vna + __builtin_convertvector((uint32x4_t){((X)>>((sh)+0))&1u,((X)>>((sh)+1))&1u,((X)>>((sh)+2))&1u,((X)>>((sh)+3))&1u}, float32x4_t)*vtwice;
                    DOADD1(j+0,bits0,0) DOADD1(j+4,bits0,4) DOADD1(j+8,bits0,8) DOADD1(j+12,bits0,12)
                    DOADD1(j+16,bits0,16) DOADD1(j+20,bits0,20) DOADD1(j+24,bits0,24) DOADD1(j+28,bits0,28)
                    DOADD1(j+32,bits1,0) DOADD1(j+36,bits1,4) DOADD1(j+40,bits1,8) DOADD1(j+44,bits1,12)
                    DOADD1(j+48,bits1,16) DOADD1(j+52,bits1,20) DOADD1(j+56,bits1,24) DOADD1(j+60,bits1,28)
#undef DOADD1
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    uint32_t bits = brow[w];
#define DOADD1(dst, X, sh) *(float32x4_t*)(crow + dst) += vna + __builtin_convertvector((uint32x4_t){((X)>>((sh)+0))&1u,((X)>>((sh)+1))&1u,((X)>>((sh)+2))&1u,((X)>>((sh)+3))&1u}, float32x4_t)*vtwice;
                    DOADD1(j+0,bits,0) DOADD1(j+4,bits,4) DOADD1(j+8,bits,8) DOADD1(j+12,bits,12)
                    DOADD1(j+16,bits,16) DOADD1(j+20,bits,20) DOADD1(j+24,bits,24) DOADD1(j+28,bits,28)
#undef DOADD1
                }
            }
        }
    }
}
