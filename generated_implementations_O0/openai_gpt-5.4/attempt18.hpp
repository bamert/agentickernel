#pragma once

typedef unsigned int uint32_t;
typedef decltype(sizeof(0)) size_t;

typedef float float32x4_t __attribute__((vector_size(16)));
typedef unsigned int uint32x4_t __attribute__((vector_size(16)));

#define CV4(bits, sh) __builtin_convertvector((uint32x4_t){((bits)>>((sh)+0))&1u,((bits)>>((sh)+1))&1u,((bits)>>((sh)+2))&1u,((bits)>>((sh)+3))&1u}, float32x4_t)

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;
    for (size_t i = 0; i < M; ++i) {
        const float* arow = A + i * K;
        float* crow = C + i * K;

        for (size_t p = 0; p + 7 < K; p += 8) {
            const float a0 = arow[p+0], a1 = arow[p+1], a2 = arow[p+2], a3 = arow[p+3];
            const float a4 = arow[p+4], a5 = arow[p+5], a6 = arow[p+6], a7 = arow[p+7];
            const float32x4_t vbase = {-(a0+a1+a2+a3+a4+a5+a6+a7),-(a0+a1+a2+a3+a4+a5+a6+a7),-(a0+a1+a2+a3+a4+a5+a6+a7),-(a0+a1+a2+a3+a4+a5+a6+a7)};
            const float32x4_t v20 = {a0+a0,a0+a0,a0+a0,a0+a0}, v21 = {a1+a1,a1+a1,a1+a1,a1+a1};
            const float32x4_t v22 = {a2+a2,a2+a2,a2+a2,a2+a2}, v23 = {a3+a3,a3+a3,a3+a3,a3+a3};
            const float32x4_t v24 = {a4+a4,a4+a4,a4+a4,a4+a4}, v25 = {a5+a5,a5+a5,a5+a5,a5+a5};
            const float32x4_t v26 = {a6+a6,a6+a6,a6+a6,a6+a6}, v27 = {a7+a7,a7+a7,a7+a7,a7+a7};
            const uint32_t* b0 = B + (p+0) * K_ints; const uint32_t* b1 = b0 + K_ints;
            const uint32_t* b2 = b1 + K_ints; const uint32_t* b3 = b2 + K_ints;
            const uint32_t* b4 = b3 + K_ints; const uint32_t* b5 = b4 + K_ints;
            const uint32_t* b6 = b5 + K_ints; const uint32_t* b7 = b6 + K_ints;
            size_t j = 0;
            if (p == 0) {
                for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                    uint32_t x0=b0[w],x1=b1[w],x2=b2[w],x3=b3[w],x4=b4[w],x5=b5[w],x6=b6[w],x7=b7[w];
                    uint32_t y0=b0[w+1],y1=b1[w+1],y2=b2[w+1],y3=b3[w+1],y4=b4[w+1],y5=b5[w+1],y6=b6[w+1],y7=b7[w+1];
#define S(dst,Q0,Q1,Q2,Q3,Q4,Q5,Q6,Q7,sh) *(float32x4_t*)(crow+dst)=vbase+CV4(Q0,sh)*v20+CV4(Q1,sh)*v21+CV4(Q2,sh)*v22+CV4(Q3,sh)*v23+CV4(Q4,sh)*v24+CV4(Q5,sh)*v25+CV4(Q6,sh)*v26+CV4(Q7,sh)*v27;
                    S(j+0,x0,x1,x2,x3,x4,x5,x6,x7,0) S(j+4,x0,x1,x2,x3,x4,x5,x6,x7,4) S(j+8,x0,x1,x2,x3,x4,x5,x6,x7,8) S(j+12,x0,x1,x2,x3,x4,x5,x6,x7,12)
                    S(j+16,x0,x1,x2,x3,x4,x5,x6,x7,16) S(j+20,x0,x1,x2,x3,x4,x5,x6,x7,20) S(j+24,x0,x1,x2,x3,x4,x5,x6,x7,24) S(j+28,x0,x1,x2,x3,x4,x5,x6,x7,28)
                    S(j+32,y0,y1,y2,y3,y4,y5,y6,y7,0) S(j+36,y0,y1,y2,y3,y4,y5,y6,y7,4) S(j+40,y0,y1,y2,y3,y4,y5,y6,y7,8) S(j+44,y0,y1,y2,y3,y4,y5,y6,y7,12)
                    S(j+48,y0,y1,y2,y3,y4,y5,y6,y7,16) S(j+52,y0,y1,y2,y3,y4,y5,y6,y7,20) S(j+56,y0,y1,y2,y3,y4,y5,y6,y7,24) S(j+60,y0,y1,y2,y3,y4,y5,y6,y7,28)
#undef S
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    uint32_t x0=b0[w],x1=b1[w],x2=b2[w],x3=b3[w],x4=b4[w],x5=b5[w],x6=b6[w],x7=b7[w];
#define S(dst,Q0,Q1,Q2,Q3,Q4,Q5,Q6,Q7,sh) *(float32x4_t*)(crow+dst)=vbase+CV4(Q0,sh)*v20+CV4(Q1,sh)*v21+CV4(Q2,sh)*v22+CV4(Q3,sh)*v23+CV4(Q4,sh)*v24+CV4(Q5,sh)*v25+CV4(Q6,sh)*v26+CV4(Q7,sh)*v27;
                    S(j+0,x0,x1,x2,x3,x4,x5,x6,x7,0) S(j+4,x0,x1,x2,x3,x4,x5,x6,x7,4) S(j+8,x0,x1,x2,x3,x4,x5,x6,x7,8) S(j+12,x0,x1,x2,x3,x4,x5,x6,x7,12)
                    S(j+16,x0,x1,x2,x3,x4,x5,x6,x7,16) S(j+20,x0,x1,x2,x3,x4,x5,x6,x7,20) S(j+24,x0,x1,x2,x3,x4,x5,x6,x7,24) S(j+28,x0,x1,x2,x3,x4,x5,x6,x7,28)
#undef S
                }
            } else {
                for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                    uint32_t x0=b0[w],x1=b1[w],x2=b2[w],x3=b3[w],x4=b4[w],x5=b5[w],x6=b6[w],x7=b7[w];
                    uint32_t y0=b0[w+1],y1=b1[w+1],y2=b2[w+1],y3=b3[w+1],y4=b4[w+1],y5=b5[w+1],y6=b6[w+1],y7=b7[w+1];
#define A1(dst,Q0,Q1,Q2,Q3,Q4,Q5,Q6,Q7,sh) *(float32x4_t*)(crow+dst)+=vbase+CV4(Q0,sh)*v20+CV4(Q1,sh)*v21+CV4(Q2,sh)*v22+CV4(Q3,sh)*v23+CV4(Q4,sh)*v24+CV4(Q5,sh)*v25+CV4(Q6,sh)*v26+CV4(Q7,sh)*v27;
                    A1(j+0,x0,x1,x2,x3,x4,x5,x6,x7,0) A1(j+4,x0,x1,x2,x3,x4,x5,x6,x7,4) A1(j+8,x0,x1,x2,x3,x4,x5,x6,x7,8) A1(j+12,x0,x1,x2,x3,x4,x5,x6,x7,12)
                    A1(j+16,x0,x1,x2,x3,x4,x5,x6,x7,16) A1(j+20,x0,x1,x2,x3,x4,x5,x6,x7,20) A1(j+24,x0,x1,x2,x3,x4,x5,x6,x7,24) A1(j+28,x0,x1,x2,x3,x4,x5,x6,x7,28)
                    A1(j+32,y0,y1,y2,y3,y4,y5,y6,y7,0) A1(j+36,y0,y1,y2,y3,y4,y5,y6,y7,4) A1(j+40,y0,y1,y2,y3,y4,y5,y6,y7,8) A1(j+44,y0,y1,y2,y3,y4,y5,y6,y7,12)
                    A1(j+48,y0,y1,y2,y3,y4,y5,y6,y7,16) A1(j+52,y0,y1,y2,y3,y4,y5,y6,y7,20) A1(j+56,y0,y1,y2,y3,y4,y5,y6,y7,24) A1(j+60,y0,y1,y2,y3,y4,y5,y6,y7,28)
#undef A1
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    uint32_t x0=b0[w],x1=b1[w],x2=b2[w],x3=b3[w],x4=b4[w],x5=b5[w],x6=b6[w],x7=b7[w];
#define A1(dst,Q0,Q1,Q2,Q3,Q4,Q5,Q6,Q7,sh) *(float32x4_t*)(crow+dst)+=vbase+CV4(Q0,sh)*v20+CV4(Q1,sh)*v21+CV4(Q2,sh)*v22+CV4(Q3,sh)*v23+CV4(Q4,sh)*v24+CV4(Q5,sh)*v25+CV4(Q6,sh)*v26+CV4(Q7,sh)*v27;
                    A1(j+0,x0,x1,x2,x3,x4,x5,x6,x7,0) A1(j+4,x0,x1,x2,x3,x4,x5,x6,x7,4) A1(j+8,x0,x1,x2,x3,x4,x5,x6,x7,8) A1(j+12,x0,x1,x2,x3,x4,x5,x6,x7,12)
                    A1(j+16,x0,x1,x2,x3,x4,x5,x6,x7,16) A1(j+20,x0,x1,x2,x3,x4,x5,x6,x7,20) A1(j+24,x0,x1,x2,x3,x4,x5,x6,x7,24) A1(j+28,x0,x1,x2,x3,x4,x5,x6,x7,28)
#undef A1
                }
            }
        }

        for (size_t p = K & ~size_t(7); p < K; ++p) {
            const float a = arow[p];
            const float32x4_t vna = {-a,-a,-a,-a};
            const float32x4_t vtwice = {a+a,a+a,a+a,a+a};
            const uint32_t* brow = B + p * K_ints;
            size_t j = 0;
            if (p == 0) {
                for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                    uint32_t q0=brow[w], q1=brow[w+1];
#define S2(dst,Q,sh) *(float32x4_t*)(crow+dst)=vna+CV4(Q,sh)*vtwice;
                    S2(j+0,q0,0) S2(j+4,q0,4) S2(j+8,q0,8) S2(j+12,q0,12) S2(j+16,q0,16) S2(j+20,q0,20) S2(j+24,q0,24) S2(j+28,q0,28)
                    S2(j+32,q1,0) S2(j+36,q1,4) S2(j+40,q1,8) S2(j+44,q1,12) S2(j+48,q1,16) S2(j+52,q1,20) S2(j+56,q1,24) S2(j+60,q1,28)
#undef S2
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    uint32_t q=brow[w];
#define S2(dst,Q,sh) *(float32x4_t*)(crow+dst)=vna+CV4(Q,sh)*vtwice;
                    S2(j+0,q,0) S2(j+4,q,4) S2(j+8,q,8) S2(j+12,q,12) S2(j+16,q,16) S2(j+20,q,20) S2(j+24,q,24) S2(j+28,q,28)
#undef S2
                }
            } else {
                for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                    uint32_t q0=brow[w], q1=brow[w+1];
#define A2(dst,Q,sh) *(float32x4_t*)(crow+dst)+=vna+CV4(Q,sh)*vtwice;
                    A2(j+0,q0,0) A2(j+4,q0,4) A2(j+8,q0,8) A2(j+12,q0,12) A2(j+16,q0,16) A2(j+20,q0,20) A2(j+24,q0,24) A2(j+28,q0,28)
                    A2(j+32,q1,0) A2(j+36,q1,4) A2(j+40,q1,8) A2(j+44,q1,12) A2(j+48,q1,16) A2(j+52,q1,20) A2(j+56,q1,24) A2(j+60,q1,28)
#undef A2
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    uint32_t q=brow[w];
#define A2(dst,Q,sh) *(float32x4_t*)(crow+dst)+=vna+CV4(Q,sh)*vtwice;
                    A2(j+0,q,0) A2(j+4,q,4) A2(j+8,q,8) A2(j+12,q,12) A2(j+16,q,16) A2(j+20,q,20) A2(j+24,q,24) A2(j+28,q,28)
#undef A2
                }
            }
        }
    }
}
