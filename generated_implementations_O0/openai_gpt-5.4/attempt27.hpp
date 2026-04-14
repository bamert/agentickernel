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
        for (size_t p = 0; p + 4 < K; p += 5) {
            const float a0 = arow[p + 0], a1 = arow[p + 1], a2 = arow[p + 2], a3 = arow[p + 3], a4 = arow[p + 4];
            const float32x4_t vb = {-(a0+a1+a2+a3+a4), -(a0+a1+a2+a3+a4), -(a0+a1+a2+a3+a4), -(a0+a1+a2+a3+a4)};
            const float32x4_t v0 = {a0+a0, a0+a0, a0+a0, a0+a0};
            const float32x4_t v1 = {a1+a1, a1+a1, a1+a1, a1+a1};
            const float32x4_t v2 = {a2+a2, a2+a2, a2+a2, a2+a2};
            const float32x4_t v3 = {a3+a3, a3+a3, a3+a3, a3+a3};
            const float32x4_t v4 = {a4+a4, a4+a4, a4+a4, a4+a4};
            const uint32_t* b0 = B + (p + 0) * K_ints;
            const uint32_t* b1 = b0 + K_ints;
            const uint32_t* b2 = b1 + K_ints;
            const uint32_t* b3 = b2 + K_ints;
            const uint32_t* b4 = b3 + K_ints;
            size_t j = 0;
            if (p == 0) {
                for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                    const uint32_t x0 = b0[w], x1 = b1[w], x2 = b2[w], x3 = b3[w], x4 = b4[w];
                    const uint32_t y0 = b0[w+1], y1 = b1[w+1], y2 = b2[w+1], y3 = b3[w+1], y4 = b4[w+1];
#define VEC(bits, sh) __builtin_convertvector((uint32x4_t){((bits)>>((sh)+0))&1u,((bits)>>((sh)+1))&1u,((bits)>>((sh)+2))&1u,((bits)>>((sh)+3))&1u}, float32x4_t)
                    *(float32x4_t*)(crow+j+0)  = vb + VEC(x0,0)*v0  + VEC(x1,0)*v1  + VEC(x2,0)*v2  + VEC(x3,0)*v3  + VEC(x4,0)*v4;
                    *(float32x4_t*)(crow+j+4)  = vb + VEC(x0,4)*v0  + VEC(x1,4)*v1  + VEC(x2,4)*v2  + VEC(x3,4)*v3  + VEC(x4,4)*v4;
                    *(float32x4_t*)(crow+j+8)  = vb + VEC(x0,8)*v0  + VEC(x1,8)*v1  + VEC(x2,8)*v2  + VEC(x3,8)*v3  + VEC(x4,8)*v4;
                    *(float32x4_t*)(crow+j+12) = vb + VEC(x0,12)*v0 + VEC(x1,12)*v1 + VEC(x2,12)*v2 + VEC(x3,12)*v3 + VEC(x4,12)*v4;
                    *(float32x4_t*)(crow+j+16) = vb + VEC(x0,16)*v0 + VEC(x1,16)*v1 + VEC(x2,16)*v2 + VEC(x3,16)*v3 + VEC(x4,16)*v4;
                    *(float32x4_t*)(crow+j+20) = vb + VEC(x0,20)*v0 + VEC(x1,20)*v1 + VEC(x2,20)*v2 + VEC(x3,20)*v3 + VEC(x4,20)*v4;
                    *(float32x4_t*)(crow+j+24) = vb + VEC(x0,24)*v0 + VEC(x1,24)*v1 + VEC(x2,24)*v2 + VEC(x3,24)*v3 + VEC(x4,24)*v4;
                    *(float32x4_t*)(crow+j+28) = vb + VEC(x0,28)*v0 + VEC(x1,28)*v1 + VEC(x2,28)*v2 + VEC(x3,28)*v3 + VEC(x4,28)*v4;
                    *(float32x4_t*)(crow+j+32) = vb + VEC(y0,0)*v0  + VEC(y1,0)*v1  + VEC(y2,0)*v2  + VEC(y3,0)*v3  + VEC(y4,0)*v4;
                    *(float32x4_t*)(crow+j+36) = vb + VEC(y0,4)*v0  + VEC(y1,4)*v1  + VEC(y2,4)*v2  + VEC(y3,4)*v3  + VEC(y4,4)*v4;
                    *(float32x4_t*)(crow+j+40) = vb + VEC(y0,8)*v0  + VEC(y1,8)*v1  + VEC(y2,8)*v2  + VEC(y3,8)*v3  + VEC(y4,8)*v4;
                    *(float32x4_t*)(crow+j+44) = vb + VEC(y0,12)*v0 + VEC(y1,12)*v1 + VEC(y2,12)*v2 + VEC(y3,12)*v3 + VEC(y4,12)*v4;
                    *(float32x4_t*)(crow+j+48) = vb + VEC(y0,16)*v0 + VEC(y1,16)*v1 + VEC(y2,16)*v2 + VEC(y3,16)*v3 + VEC(y4,16)*v4;
                    *(float32x4_t*)(crow+j+52) = vb + VEC(y0,20)*v0 + VEC(y1,20)*v1 + VEC(y2,20)*v2 + VEC(y3,20)*v3 + VEC(y4,20)*v4;
                    *(float32x4_t*)(crow+j+56) = vb + VEC(y0,24)*v0 + VEC(y1,24)*v1 + VEC(y2,24)*v2 + VEC(y3,24)*v3 + VEC(y4,24)*v4;
                    *(float32x4_t*)(crow+j+60) = vb + VEC(y0,28)*v0 + VEC(y1,28)*v1 + VEC(y2,28)*v2 + VEC(y3,28)*v3 + VEC(y4,28)*v4;
#undef VEC
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    const uint32_t x0 = b0[w], x1 = b1[w], x2 = b2[w], x3 = b3[w], x4 = b4[w];
#define VEC(bits, sh) __builtin_convertvector((uint32x4_t){((bits)>>((sh)+0))&1u,((bits)>>((sh)+1))&1u,((bits)>>((sh)+2))&1u,((bits)>>((sh)+3))&1u}, float32x4_t)
                    *(float32x4_t*)(crow+j+0)  = vb + VEC(x0,0)*v0  + VEC(x1,0)*v1  + VEC(x2,0)*v2  + VEC(x3,0)*v3  + VEC(x4,0)*v4;
                    *(float32x4_t*)(crow+j+4)  = vb + VEC(x0,4)*v0  + VEC(x1,4)*v1  + VEC(x2,4)*v2  + VEC(x3,4)*v3  + VEC(x4,4)*v4;
                    *(float32x4_t*)(crow+j+8)  = vb + VEC(x0,8)*v0  + VEC(x1,8)*v1  + VEC(x2,8)*v2  + VEC(x3,8)*v3  + VEC(x4,8)*v4;
                    *(float32x4_t*)(crow+j+12) = vb + VEC(x0,12)*v0 + VEC(x1,12)*v1 + VEC(x2,12)*v2 + VEC(x3,12)*v3 + VEC(x4,12)*v4;
                    *(float32x4_t*)(crow+j+16) = vb + VEC(x0,16)*v0 + VEC(x1,16)*v1 + VEC(x2,16)*v2 + VEC(x3,16)*v3 + VEC(x4,16)*v4;
                    *(float32x4_t*)(crow+j+20) = vb + VEC(x0,20)*v0 + VEC(x1,20)*v1 + VEC(x2,20)*v2 + VEC(x3,20)*v3 + VEC(x4,20)*v4;
                    *(float32x4_t*)(crow+j+24) = vb + VEC(x0,24)*v0 + VEC(x1,24)*v1 + VEC(x2,24)*v2 + VEC(x3,24)*v3 + VEC(x4,24)*v4;
                    *(float32x4_t*)(crow+j+28) = vb + VEC(x0,28)*v0 + VEC(x1,28)*v1 + VEC(x2,28)*v2 + VEC(x3,28)*v3 + VEC(x4,28)*v4;
#undef VEC
                }
            } else {
                for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                    const uint32_t x0 = b0[w], x1 = b1[w], x2 = b2[w], x3 = b3[w], x4 = b4[w];
                    const uint32_t y0 = b0[w+1], y1 = b1[w+1], y2 = b2[w+1], y3 = b3[w+1], y4 = b4[w+1];
#define VEC(bits, sh) __builtin_convertvector((uint32x4_t){((bits)>>((sh)+0))&1u,((bits)>>((sh)+1))&1u,((bits)>>((sh)+2))&1u,((bits)>>((sh)+3))&1u}, float32x4_t)
                    *(float32x4_t*)(crow+j+0)  += vb + VEC(x0,0)*v0  + VEC(x1,0)*v1  + VEC(x2,0)*v2  + VEC(x3,0)*v3  + VEC(x4,0)*v4;
                    *(float32x4_t*)(crow+j+4)  += vb + VEC(x0,4)*v0  + VEC(x1,4)*v1  + VEC(x2,4)*v2  + VEC(x3,4)*v3  + VEC(x4,4)*v4;
                    *(float32x4_t*)(crow+j+8)  += vb + VEC(x0,8)*v0  + VEC(x1,8)*v1  + VEC(x2,8)*v2  + VEC(x3,8)*v3  + VEC(x4,8)*v4;
                    *(float32x4_t*)(crow+j+12) += vb + VEC(x0,12)*v0 + VEC(x1,12)*v1 + VEC(x2,12)*v2 + VEC(x3,12)*v3 + VEC(x4,12)*v4;
                    *(float32x4_t*)(crow+j+16) += vb + VEC(x0,16)*v0 + VEC(x1,16)*v1 + VEC(x2,16)*v2 + VEC(x3,16)*v3 + VEC(x4,16)*v4;
                    *(float32x4_t*)(crow+j+20) += vb + VEC(x0,20)*v0 + VEC(x1,20)*v1 + VEC(x2,20)*v2 + VEC(x3,20)*v3 + VEC(x4,20)*v4;
                    *(float32x4_t*)(crow+j+24) += vb + VEC(x0,24)*v0 + VEC(x1,24)*v1 + VEC(x2,24)*v2 + VEC(x3,24)*v3 + VEC(x4,24)*v4;
                    *(float32x4_t*)(crow+j+28) += vb + VEC(x0,28)*v0 + VEC(x1,28)*v1 + VEC(x2,28)*v2 + VEC(x3,28)*v3 + VEC(x4,28)*v4;
                    *(float32x4_t*)(crow+j+32) += vb + VEC(y0,0)*v0  + VEC(y1,0)*v1  + VEC(y2,0)*v2  + VEC(y3,0)*v3  + VEC(y4,0)*v4;
                    *(float32x4_t*)(crow+j+36) += vb + VEC(y0,4)*v0  + VEC(y1,4)*v1  + VEC(y2,4)*v2  + VEC(y3,4)*v3  + VEC(y4,4)*v4;
                    *(float32x4_t*)(crow+j+40) += vb + VEC(y0,8)*v0  + VEC(y1,8)*v1  + VEC(y2,8)*v2  + VEC(y3,8)*v3  + VEC(y4,8)*v4;
                    *(float32x4_t*)(crow+j+44) += vb + VEC(y0,12)*v0 + VEC(y1,12)*v1 + VEC(y2,12)*v2 + VEC(y3,12)*v3 + VEC(y4,12)*v4;
                    *(float32x4_t*)(crow+j+48) += vb + VEC(y0,16)*v0 + VEC(y1,16)*v1 + VEC(y2,16)*v2 + VEC(y3,16)*v3 + VEC(y4,16)*v4;
                    *(float32x4_t*)(crow+j+52) += vb + VEC(y0,20)*v0 + VEC(y1,20)*v1 + VEC(y2,20)*v2 + VEC(y3,20)*v3 + VEC(y4,20)*v4;
                    *(float32x4_t*)(crow+j+56) += vb + VEC(y0,24)*v0 + VEC(y1,24)*v1 + VEC(y2,24)*v2 + VEC(y3,24)*v3 + VEC(y4,24)*v4;
                    *(float32x4_t*)(crow+j+60) += vb + VEC(y0,28)*v0 + VEC(y1,28)*v1 + VEC(y2,28)*v2 + VEC(y3,28)*v3 + VEC(y4,28)*v4;
#undef VEC
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    const uint32_t x0 = b0[w], x1 = b1[w], x2 = b2[w], x3 = b3[w], x4 = b4[w];
#define VEC(bits, sh) __builtin_convertvector((uint32x4_t){((bits)>>((sh)+0))&1u,((bits)>>((sh)+1))&1u,((bits)>>((sh)+2))&1u,((bits)>>((sh)+3))&1u}, float32x4_t)
                    *(float32x4_t*)(crow+j+0)  += vb + VEC(x0,0)*v0  + VEC(x1,0)*v1  + VEC(x2,0)*v2  + VEC(x3,0)*v3  + VEC(x4,0)*v4;
                    *(float32x4_t*)(crow+j+4)  += vb + VEC(x0,4)*v0  + VEC(x1,4)*v1  + VEC(x2,4)*v2  + VEC(x3,4)*v3  + VEC(x4,4)*v4;
                    *(float32x4_t*)(crow+j+8)  += vb + VEC(x0,8)*v0  + VEC(x1,8)*v1  + VEC(x2,8)*v2  + VEC(x3,8)*v3  + VEC(x4,8)*v4;
                    *(float32x4_t*)(crow+j+12) += vb + VEC(x0,12)*v0 + VEC(x1,12)*v1 + VEC(x2,12)*v2 + VEC(x3,12)*v3 + VEC(x4,12)*v4;
                    *(float32x4_t*)(crow+j+16) += vb + VEC(x0,16)*v0 + VEC(x1,16)*v1 + VEC(x2,16)*v2 + VEC(x3,16)*v3 + VEC(x4,16)*v4;
                    *(float32x4_t*)(crow+j+20) += vb + VEC(x0,20)*v0 + VEC(x1,20)*v1 + VEC(x2,20)*v2 + VEC(x3,20)*v3 + VEC(x4,20)*v4;
                    *(float32x4_t*)(crow+j+24) += vb + VEC(x0,24)*v0 + VEC(x1,24)*v1 + VEC(x2,24)*v2 + VEC(x3,24)*v3 + VEC(x4,24)*v4;
                    *(float32x4_t*)(crow+j+28) += vb + VEC(x0,28)*v0 + VEC(x1,28)*v1 + VEC(x2,28)*v2 + VEC(x3,28)*v3 + VEC(x4,28)*v4;
#undef VEC
                }
            }
        }
        for (size_t p = (K/5)*5; p < K; ++p) {
            const float a = arow[p];
            const float32x4_t vna = {-a,-a,-a,-a};
            const float32x4_t vtw = {a+a,a+a,a+a,a+a};
            const uint32_t* brow = B + p * K_ints;
            size_t j = 0;
            if (p == 0) {
                for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                    const uint32_t q0 = brow[w], q1 = brow[w+1];
#define VEC(bits, sh) __builtin_convertvector((uint32x4_t){((bits)>>((sh)+0))&1u,((bits)>>((sh)+1))&1u,((bits)>>((sh)+2))&1u,((bits)>>((sh)+3))&1u}, float32x4_t)
                    *(float32x4_t*)(crow+j+0)  = vna + VEC(q0,0)*vtw;  *(float32x4_t*)(crow+j+4)  = vna + VEC(q0,4)*vtw;
                    *(float32x4_t*)(crow+j+8)  = vna + VEC(q0,8)*vtw;  *(float32x4_t*)(crow+j+12) = vna + VEC(q0,12)*vtw;
                    *(float32x4_t*)(crow+j+16) = vna + VEC(q0,16)*vtw; *(float32x4_t*)(crow+j+20) = vna + VEC(q0,20)*vtw;
                    *(float32x4_t*)(crow+j+24) = vna + VEC(q0,24)*vtw; *(float32x4_t*)(crow+j+28) = vna + VEC(q0,28)*vtw;
                    *(float32x4_t*)(crow+j+32) = vna + VEC(q1,0)*vtw;  *(float32x4_t*)(crow+j+36) = vna + VEC(q1,4)*vtw;
                    *(float32x4_t*)(crow+j+40) = vna + VEC(q1,8)*vtw;  *(float32x4_t*)(crow+j+44) = vna + VEC(q1,12)*vtw;
                    *(float32x4_t*)(crow+j+48) = vna + VEC(q1,16)*vtw; *(float32x4_t*)(crow+j+52) = vna + VEC(q1,20)*vtw;
                    *(float32x4_t*)(crow+j+56) = vna + VEC(q1,24)*vtw; *(float32x4_t*)(crow+j+60) = vna + VEC(q1,28)*vtw;
#undef VEC
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    const uint32_t q = brow[w];
#define VEC(bits, sh) __builtin_convertvector((uint32x4_t){((bits)>>((sh)+0))&1u,((bits)>>((sh)+1))&1u,((bits)>>((sh)+2))&1u,((bits)>>((sh)+3))&1u}, float32x4_t)
                    *(float32x4_t*)(crow+j+0)  = vna + VEC(q,0)*vtw;  *(float32x4_t*)(crow+j+4)  = vna + VEC(q,4)*vtw;
                    *(float32x4_t*)(crow+j+8)  = vna + VEC(q,8)*vtw;  *(float32x4_t*)(crow+j+12) = vna + VEC(q,12)*vtw;
                    *(float32x4_t*)(crow+j+16) = vna + VEC(q,16)*vtw; *(float32x4_t*)(crow+j+20) = vna + VEC(q,20)*vtw;
                    *(float32x4_t*)(crow+j+24) = vna + VEC(q,24)*vtw; *(float32x4_t*)(crow+j+28) = vna + VEC(q,28)*vtw;
#undef VEC
                }
            } else {
                for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                    const uint32_t q0 = brow[w], q1 = brow[w+1];
#define VEC(bits, sh) __builtin_convertvector((uint32x4_t){((bits)>>((sh)+0))&1u,((bits)>>((sh)+1))&1u,((bits)>>((sh)+2))&1u,((bits)>>((sh)+3))&1u}, float32x4_t)
                    *(float32x4_t*)(crow+j+0)  += vna + VEC(q0,0)*vtw;  *(float32x4_t*)(crow+j+4)  += vna + VEC(q0,4)*vtw;
                    *(float32x4_t*)(crow+j+8)  += vna + VEC(q0,8)*vtw;  *(float32x4_t*)(crow+j+12) += vna + VEC(q0,12)*vtw;
                    *(float32x4_t*)(crow+j+16) += vna + VEC(q0,16)*vtw; *(float32x4_t*)(crow+j+20) += vna + VEC(q0,20)*vtw;
                    *(float32x4_t*)(crow+j+24) += vna + VEC(q0,24)*vtw; *(float32x4_t*)(crow+j+28) += vna + VEC(q0,28)*vtw;
                    *(float32x4_t*)(crow+j+32) += vna + VEC(q1,0)*vtw;  *(float32x4_t*)(crow+j+36) += vna + VEC(q1,4)*vtw;
                    *(float32x4_t*)(crow+j+40) += vna + VEC(q1,8)*vtw;  *(float32x4_t*)(crow+j+44) += vna + VEC(q1,12)*vtw;
                    *(float32x4_t*)(crow+j+48) += vna + VEC(q1,16)*vtw; *(float32x4_t*)(crow+j+52) += vna + VEC(q1,20)*vtw;
                    *(float32x4_t*)(crow+j+56) += vna + VEC(q1,24)*vtw; *(float32x4_t*)(crow+j+60) += vna + VEC(q1,28)*vtw;
#undef VEC
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    const uint32_t q = brow[w];
#define VEC(bits, sh) __builtin_convertvector((uint32x4_t){((bits)>>((sh)+0))&1u,((bits)>>((sh)+1))&1u,((bits)>>((sh)+2))&1u,((bits)>>((sh)+3))&1u}, float32x4_t)
                    *(float32x4_t*)(crow+j+0)  += vna + VEC(q,0)*vtw;  *(float32x4_t*)(crow+j+4)  += vna + VEC(q,4)*vtw;
                    *(float32x4_t*)(crow+j+8)  += vna + VEC(q,8)*vtw;  *(float32x4_t*)(crow+j+12) += vna + VEC(q,12)*vtw;
                    *(float32x4_t*)(crow+j+16) += vna + VEC(q,16)*vtw; *(float32x4_t*)(crow+j+20) += vna + VEC(q,20)*vtw;
                    *(float32x4_t*)(crow+j+24) += vna + VEC(q,24)*vtw; *(float32x4_t*)(crow+j+28) += vna + VEC(q,28)*vtw;
#undef VEC
                }
            }
        }
    }
}
