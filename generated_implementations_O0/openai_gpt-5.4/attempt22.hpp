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
        for (size_t p = 0; p + 4 < K; p += 5) {
            const float a0=arow[p+0], a1=arow[p+1], a2=arow[p+2], a3=arow[p+3], a4=arow[p+4];
            const float32x4_t vb={-(a0+a1+a2+a3+a4),-(a0+a1+a2+a3+a4),-(a0+a1+a2+a3+a4),-(a0+a1+a2+a3+a4)};
            const float32x4_t v0={a0+a0,a0+a0,a0+a0,a0+a0}, v1={a1+a1,a1+a1,a1+a1,a1+a1}, v2={a2+a2,a2+a2,a2+a2,a2+a2};
            const float32x4_t v3={a3+a3,a3+a3,a3+a3,a3+a3}, v4={a4+a4,a4+a4,a4+a4,a4+a4};
            const uint32_t* b0=B+(p+0)*K_ints; const uint32_t* b1=b0+K_ints; const uint32_t* b2=b1+K_ints;
            const uint32_t* b3=b2+K_ints; const uint32_t* b4=b3+K_ints;
            size_t j=0;
            if (p==0) {
                for (size_t w=0; w+3<K_ints; w+=4, j+=128) {
                    uint32_t x0=b0[w],x1=b1[w],x2=b2[w],x3=b3[w],x4=b4[w];
                    uint32_t y0=b0[w+1],y1=b1[w+1],y2=b2[w+1],y3=b3[w+1],y4=b4[w+1];
                    uint32_t z0=b0[w+2],z1=b1[w+2],z2=b2[w+2],z3=b3[w+2],z4=b4[w+2];
                    uint32_t t0=b0[w+3],t1=b1[w+3],t2=b2[w+3],t3=b3[w+3],t4=b4[w+3];
#define S(dst,Q0,Q1,Q2,Q3,Q4,sh) *(float32x4_t*)(crow+dst)=vb+CV4(Q0,sh)*v0+CV4(Q1,sh)*v1+CV4(Q2,sh)*v2+CV4(Q3,sh)*v3+CV4(Q4,sh)*v4;
                    S(j+0,x0,x1,x2,x3,x4,0) S(j+4,x0,x1,x2,x3,x4,4) S(j+8,x0,x1,x2,x3,x4,8) S(j+12,x0,x1,x2,x3,x4,12)
                    S(j+16,x0,x1,x2,x3,x4,16) S(j+20,x0,x1,x2,x3,x4,20) S(j+24,x0,x1,x2,x3,x4,24) S(j+28,x0,x1,x2,x3,x4,28)
                    S(j+32,y0,y1,y2,y3,y4,0) S(j+36,y0,y1,y2,y3,y4,4) S(j+40,y0,y1,y2,y3,y4,8) S(j+44,y0,y1,y2,y3,y4,12)
                    S(j+48,y0,y1,y2,y3,y4,16) S(j+52,y0,y1,y2,y3,y4,20) S(j+56,y0,y1,y2,y3,y4,24) S(j+60,y0,y1,y2,y3,y4,28)
                    S(j+64,z0,z1,z2,z3,z4,0) S(j+68,z0,z1,z2,z3,z4,4) S(j+72,z0,z1,z2,z3,z4,8) S(j+76,z0,z1,z2,z3,z4,12)
                    S(j+80,z0,z1,z2,z3,z4,16) S(j+84,z0,z1,z2,z3,z4,20) S(j+88,z0,z1,z2,z3,z4,24) S(j+92,z0,z1,z2,z3,z4,28)
                    S(j+96,t0,t1,t2,t3,t4,0) S(j+100,t0,t1,t2,t3,t4,4) S(j+104,t0,t1,t2,t3,t4,8) S(j+108,t0,t1,t2,t3,t4,12)
                    S(j+112,t0,t1,t2,t3,t4,16) S(j+116,t0,t1,t2,t3,t4,20) S(j+120,t0,t1,t2,t3,t4,24) S(j+124,t0,t1,t2,t3,t4,28)
#undef S
                }
                for (size_t w=(K_ints & ~size_t(3)); w<K_ints; ++w, j+=32) {
                    uint32_t x0=b0[w],x1=b1[w],x2=b2[w],x3=b3[w],x4=b4[w];
#define S(dst,Q0,Q1,Q2,Q3,Q4,sh) *(float32x4_t*)(crow+dst)=vb+CV4(Q0,sh)*v0+CV4(Q1,sh)*v1+CV4(Q2,sh)*v2+CV4(Q3,sh)*v3+CV4(Q4,sh)*v4;
                    S(j+0,x0,x1,x2,x3,x4,0) S(j+4,x0,x1,x2,x3,x4,4) S(j+8,x0,x1,x2,x3,x4,8) S(j+12,x0,x1,x2,x3,x4,12)
                    S(j+16,x0,x1,x2,x3,x4,16) S(j+20,x0,x1,x2,x3,x4,20) S(j+24,x0,x1,x2,x3,x4,24) S(j+28,x0,x1,x2,x3,x4,28)
#undef S
                }
            } else {
                for (size_t w=0; w+3<K_ints; w+=4, j+=128) {
                    uint32_t x0=b0[w],x1=b1[w],x2=b2[w],x3=b3[w],x4=b4[w];
                    uint32_t y0=b0[w+1],y1=b1[w+1],y2=b2[w+1],y3=b3[w+1],y4=b4[w+1];
                    uint32_t z0=b0[w+2],z1=b1[w+2],z2=b2[w+2],z3=b3[w+2],z4=b4[w+2];
                    uint32_t t0=b0[w+3],t1=b1[w+3],t2=b2[w+3],t3=b3[w+3],t4=b4[w+3];
#define A(dst,Q0,Q1,Q2,Q3,Q4,sh) *(float32x4_t*)(crow+dst)+=vb+CV4(Q0,sh)*v0+CV4(Q1,sh)*v1+CV4(Q2,sh)*v2+CV4(Q3,sh)*v3+CV4(Q4,sh)*v4;
                    A(j+0,x0,x1,x2,x3,x4,0) A(j+4,x0,x1,x2,x3,x4,4) A(j+8,x0,x1,x2,x3,x4,8) A(j+12,x0,x1,x2,x3,x4,12)
                    A(j+16,x0,x1,x2,x3,x4,16) A(j+20,x0,x1,x2,x3,x4,20) A(j+24,x0,x1,x2,x3,x4,24) A(j+28,x0,x1,x2,x3,x4,28)
                    A(j+32,y0,y1,y2,y3,y4,0) A(j+36,y0,y1,y2,y3,y4,4) A(j+40,y0,y1,y2,y3,y4,8) A(j+44,y0,y1,y2,y3,y4,12)
                    A(j+48,y0,y1,y2,y3,y4,16) A(j+52,y0,y1,y2,y3,y4,20) A(j+56,y0,y1,y2,y3,y4,24) A(j+60,y0,y1,y2,y3,y4,28)
                    A(j+64,z0,z1,z2,z3,z4,0) A(j+68,z0,z1,z2,z3,z4,4) A(j+72,z0,z1,z2,z3,z4,8) A(j+76,z0,z1,z2,z3,z4,12)
                    A(j+80,z0,z1,z2,z3,z4,16) A(j+84,z0,z1,z2,z3,z4,20) A(j+88,z0,z1,z2,z3,z4,24) A(j+92,z0,z1,z2,z3,z4,28)
                    A(j+96,t0,t1,t2,t3,t4,0) A(j+100,t0,t1,t2,t3,t4,4) A(j+104,t0,t1,t2,t3,t4,8) A(j+108,t0,t1,t2,t3,t4,12)
                    A(j+112,t0,t1,t2,t3,t4,16) A(j+116,t0,t1,t2,t3,t4,20) A(j+120,t0,t1,t2,t3,t4,24) A(j+124,t0,t1,t2,t3,t4,28)
#undef A
                }
                for (size_t w=(K_ints & ~size_t(3)); w<K_ints; ++w, j+=32) {
                    uint32_t x0=b0[w],x1=b1[w],x2=b2[w],x3=b3[w],x4=b4[w];
#define A(dst,Q0,Q1,Q2,Q3,Q4,sh) *(float32x4_t*)(crow+dst)+=vb+CV4(Q0,sh)*v0+CV4(Q1,sh)*v1+CV4(Q2,sh)*v2+CV4(Q3,sh)*v3+CV4(Q4,sh)*v4;
                    A(j+0,x0,x1,x2,x3,x4,0) A(j+4,x0,x1,x2,x3,x4,4) A(j+8,x0,x1,x2,x3,x4,8) A(j+12,x0,x1,x2,x3,x4,12)
                    A(j+16,x0,x1,x2,x3,x4,16) A(j+20,x0,x1,x2,x3,x4,20) A(j+24,x0,x1,x2,x3,x4,24) A(j+28,x0,x1,x2,x3,x4,28)
#undef A
                }
            }
        }
        for (size_t p=(K/5)*5; p<K; ++p) {
            const float a=arow[p];
            const float32x4_t vna={-a,-a,-a,-a}, vtw={a+a,a+a,a+a,a+a};
            const uint32_t* brow=B+p*K_ints;
            size_t j=0;
            if (p==0) {
                for (size_t w=0; w+3<K_ints; w+=4, j+=128) {
                    uint32_t q0=brow[w], q1=brow[w+1], q2=brow[w+2], q3=brow[w+3];
#define S1(dst,Q,sh) *(float32x4_t*)(crow+dst)=vna+CV4(Q,sh)*vtw;
                    S1(j+0,q0,0) S1(j+4,q0,4) S1(j+8,q0,8) S1(j+12,q0,12) S1(j+16,q0,16) S1(j+20,q0,20) S1(j+24,q0,24) S1(j+28,q0,28)
                    S1(j+32,q1,0) S1(j+36,q1,4) S1(j+40,q1,8) S1(j+44,q1,12) S1(j+48,q1,16) S1(j+52,q1,20) S1(j+56,q1,24) S1(j+60,q1,28)
                    S1(j+64,q2,0) S1(j+68,q2,4) S1(j+72,q2,8) S1(j+76,q2,12) S1(j+80,q2,16) S1(j+84,q2,20) S1(j+88,q2,24) S1(j+92,q2,28)
                    S1(j+96,q3,0) S1(j+100,q3,4) S1(j+104,q3,8) S1(j+108,q3,12) S1(j+112,q3,16) S1(j+116,q3,20) S1(j+120,q3,24) S1(j+124,q3,28)
#undef S1
                }
                for (size_t w=(K_ints & ~size_t(3)); w<K_ints; ++w, j+=32) {
                    uint32_t q=brow[w];
#define S1(dst,Q,sh) *(float32x4_t*)(crow+dst)=vna+CV4(Q,sh)*vtw;
                    S1(j+0,q,0) S1(j+4,q,4) S1(j+8,q,8) S1(j+12,q,12) S1(j+16,q,16) S1(j+20,q,20) S1(j+24,q,24) S1(j+28,q,28)
#undef S1
                }
            } else {
                for (size_t w=0; w+3<K_ints; w+=4, j+=128) {
                    uint32_t q0=brow[w], q1=brow[w+1], q2=brow[w+2], q3=brow[w+3];
#define A1(dst,Q,sh) *(float32x4_t*)(crow+dst)+=vna+CV4(Q,sh)*vtw;
                    A1(j+0,q0,0) A1(j+4,q0,4) A1(j+8,q0,8) A1(j+12,q0,12) A1(j+16,q0,16) A1(j+20,q0,20) A1(j+24,q0,24) A1(j+28,q0,28)
                    A1(j+32,q1,0) A1(j+36,q1,4) A1(j+40,q1,8) A1(j+44,q1,12) A1(j+48,q1,16) A1(j+52,q1,20) A1(j+56,q1,24) A1(j+60,q1,28)
                    A1(j+64,q2,0) A1(j+68,q2,4) A1(j+72,q2,8) A1(j+76,q2,12) A1(j+80,q2,16) A1(j+84,q2,20) A1(j+88,q2,24) A1(j+92,q2,28)
                    A1(j+96,q3,0) A1(j+100,q3,4) A1(j+104,q3,8) A1(j+108,q3,12) A1(j+112,q3,16) A1(j+116,q3,20) A1(j+120,q3,24) A1(j+124,q3,28)
#undef A1
                }
                for (size_t w=(K_ints & ~size_t(3)); w<K_ints; ++w, j+=32) {
                    uint32_t q=brow[w];
#define A1(dst,Q,sh) *(float32x4_t*)(crow+dst)+=vna+CV4(Q,sh)*vtw;
                    A1(j+0,q,0) A1(j+4,q,4) A1(j+8,q,8) A1(j+12,q,12) A1(j+16,q,16) A1(j+20,q,20) A1(j+24,q,24) A1(j+28,q,28)
#undef A1
                }
            }
        }
    }
}
