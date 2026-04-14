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
        for (size_t p = 0; p + 3 < K; p += 4) {
            const float a0 = arow[p+0], a1 = arow[p+1], a2 = arow[p+2], a3 = arow[p+3];
            const float32x4_t vbase = {-(a0+a1+a2+a3),-(a0+a1+a2+a3),-(a0+a1+a2+a3),-(a0+a1+a2+a3)};
            const float32x4_t v20 = {a0+a0,a0+a0,a0+a0,a0+a0};
            const float32x4_t v21 = {a1+a1,a1+a1,a1+a1,a1+a1};
            const float32x4_t v22 = {a2+a2,a2+a2,a2+a2,a2+a2};
            const float32x4_t v23 = {a3+a3,a3+a3,a3+a3,a3+a3};
            const uint32_t* b0 = B + (p+0) * K_ints;
            const uint32_t* b1 = b0 + K_ints;
            const uint32_t* b2 = b1 + K_ints;
            const uint32_t* b3 = b2 + K_ints;
            size_t j = 0;
            if (p == 0) {
                for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                    uint32_t x0=b0[w],y0=b1[w],z0=b2[w],t0=b3[w];
                    uint32_t x1=b0[w+1],y1=b1[w+1],z1=b2[w+1],t1=b3[w+1];
                    float32x4_t r0 = vbase + CV4(x0,0)*v20 + CV4(y0,0)*v21 + CV4(z0,0)*v22 + CV4(t0,0)*v23;
                    float32x4_t r1 = vbase + CV4(x0,4)*v20 + CV4(y0,4)*v21 + CV4(z0,4)*v22 + CV4(t0,4)*v23;
                    float32x4_t r2 = vbase + CV4(x0,8)*v20 + CV4(y0,8)*v21 + CV4(z0,8)*v22 + CV4(t0,8)*v23;
                    float32x4_t r3 = vbase + CV4(x0,12)*v20 + CV4(y0,12)*v21 + CV4(z0,12)*v22 + CV4(t0,12)*v23;
                    float32x4_t r4 = vbase + CV4(x0,16)*v20 + CV4(y0,16)*v21 + CV4(z0,16)*v22 + CV4(t0,16)*v23;
                    float32x4_t r5 = vbase + CV4(x0,20)*v20 + CV4(y0,20)*v21 + CV4(z0,20)*v22 + CV4(t0,20)*v23;
                    float32x4_t r6 = vbase + CV4(x0,24)*v20 + CV4(y0,24)*v21 + CV4(z0,24)*v22 + CV4(t0,24)*v23;
                    float32x4_t r7 = vbase + CV4(x0,28)*v20 + CV4(y0,28)*v21 + CV4(z0,28)*v22 + CV4(t0,28)*v23;
                    float32x4_t r8 = vbase + CV4(x1,0)*v20 + CV4(y1,0)*v21 + CV4(z1,0)*v22 + CV4(t1,0)*v23;
                    float32x4_t r9 = vbase + CV4(x1,4)*v20 + CV4(y1,4)*v21 + CV4(z1,4)*v22 + CV4(t1,4)*v23;
                    float32x4_t r10= vbase + CV4(x1,8)*v20 + CV4(y1,8)*v21 + CV4(z1,8)*v22 + CV4(t1,8)*v23;
                    float32x4_t r11= vbase + CV4(x1,12)*v20 + CV4(y1,12)*v21 + CV4(z1,12)*v22 + CV4(t1,12)*v23;
                    float32x4_t r12= vbase + CV4(x1,16)*v20 + CV4(y1,16)*v21 + CV4(z1,16)*v22 + CV4(t1,16)*v23;
                    float32x4_t r13= vbase + CV4(x1,20)*v20 + CV4(y1,20)*v21 + CV4(z1,20)*v22 + CV4(t1,20)*v23;
                    float32x4_t r14= vbase + CV4(x1,24)*v20 + CV4(y1,24)*v21 + CV4(z1,24)*v22 + CV4(t1,24)*v23;
                    float32x4_t r15= vbase + CV4(x1,28)*v20 + CV4(y1,28)*v21 + CV4(z1,28)*v22 + CV4(t1,28)*v23;
                    *(float32x4_t*)(crow+j+0)=r0; *(float32x4_t*)(crow+j+4)=r1; *(float32x4_t*)(crow+j+8)=r2; *(float32x4_t*)(crow+j+12)=r3;
                    *(float32x4_t*)(crow+j+16)=r4; *(float32x4_t*)(crow+j+20)=r5; *(float32x4_t*)(crow+j+24)=r6; *(float32x4_t*)(crow+j+28)=r7;
                    *(float32x4_t*)(crow+j+32)=r8; *(float32x4_t*)(crow+j+36)=r9; *(float32x4_t*)(crow+j+40)=r10; *(float32x4_t*)(crow+j+44)=r11;
                    *(float32x4_t*)(crow+j+48)=r12; *(float32x4_t*)(crow+j+52)=r13; *(float32x4_t*)(crow+j+56)=r14; *(float32x4_t*)(crow+j+60)=r15;
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    uint32_t x=b0[w],y=b1[w],z=b2[w],t=b3[w];
                    *(float32x4_t*)(crow+j+0)=vbase+CV4(x,0)*v20+CV4(y,0)*v21+CV4(z,0)*v22+CV4(t,0)*v23;
                    *(float32x4_t*)(crow+j+4)=vbase+CV4(x,4)*v20+CV4(y,4)*v21+CV4(z,4)*v22+CV4(t,4)*v23;
                    *(float32x4_t*)(crow+j+8)=vbase+CV4(x,8)*v20+CV4(y,8)*v21+CV4(z,8)*v22+CV4(t,8)*v23;
                    *(float32x4_t*)(crow+j+12)=vbase+CV4(x,12)*v20+CV4(y,12)*v21+CV4(z,12)*v22+CV4(t,12)*v23;
                    *(float32x4_t*)(crow+j+16)=vbase+CV4(x,16)*v20+CV4(y,16)*v21+CV4(z,16)*v22+CV4(t,16)*v23;
                    *(float32x4_t*)(crow+j+20)=vbase+CV4(x,20)*v20+CV4(y,20)*v21+CV4(z,20)*v22+CV4(t,20)*v23;
                    *(float32x4_t*)(crow+j+24)=vbase+CV4(x,24)*v20+CV4(y,24)*v21+CV4(z,24)*v22+CV4(t,24)*v23;
                    *(float32x4_t*)(crow+j+28)=vbase+CV4(x,28)*v20+CV4(y,28)*v21+CV4(z,28)*v22+CV4(t,28)*v23;
                }
            } else {
                for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                    uint32_t x0=b0[w],y0=b1[w],z0=b2[w],t0=b3[w];
                    uint32_t x1=b0[w+1],y1=b1[w+1],z1=b2[w+1],t1=b3[w+1];
                    float32x4_t r0 = vbase + CV4(x0,0)*v20 + CV4(y0,0)*v21 + CV4(z0,0)*v22 + CV4(t0,0)*v23;
                    float32x4_t r1 = vbase + CV4(x0,4)*v20 + CV4(y0,4)*v21 + CV4(z0,4)*v22 + CV4(t0,4)*v23;
                    float32x4_t r2 = vbase + CV4(x0,8)*v20 + CV4(y0,8)*v21 + CV4(z0,8)*v22 + CV4(t0,8)*v23;
                    float32x4_t r3 = vbase + CV4(x0,12)*v20 + CV4(y0,12)*v21 + CV4(z0,12)*v22 + CV4(t0,12)*v23;
                    float32x4_t r4 = vbase + CV4(x0,16)*v20 + CV4(y0,16)*v21 + CV4(z0,16)*v22 + CV4(t0,16)*v23;
                    float32x4_t r5 = vbase + CV4(x0,20)*v20 + CV4(y0,20)*v21 + CV4(z0,20)*v22 + CV4(t0,20)*v23;
                    float32x4_t r6 = vbase + CV4(x0,24)*v20 + CV4(y0,24)*v21 + CV4(z0,24)*v22 + CV4(t0,24)*v23;
                    float32x4_t r7 = vbase + CV4(x0,28)*v20 + CV4(y0,28)*v21 + CV4(z0,28)*v22 + CV4(t0,28)*v23;
                    float32x4_t r8 = vbase + CV4(x1,0)*v20 + CV4(y1,0)*v21 + CV4(z1,0)*v22 + CV4(t1,0)*v23;
                    float32x4_t r9 = vbase + CV4(x1,4)*v20 + CV4(y1,4)*v21 + CV4(z1,4)*v22 + CV4(t1,4)*v23;
                    float32x4_t r10= vbase + CV4(x1,8)*v20 + CV4(y1,8)*v21 + CV4(z1,8)*v22 + CV4(t1,8)*v23;
                    float32x4_t r11= vbase + CV4(x1,12)*v20 + CV4(y1,12)*v21 + CV4(z1,12)*v22 + CV4(t1,12)*v23;
                    float32x4_t r12= vbase + CV4(x1,16)*v20 + CV4(y1,16)*v21 + CV4(z1,16)*v22 + CV4(t1,16)*v23;
                    float32x4_t r13= vbase + CV4(x1,20)*v20 + CV4(y1,20)*v21 + CV4(z1,20)*v22 + CV4(t1,20)*v23;
                    float32x4_t r14= vbase + CV4(x1,24)*v20 + CV4(y1,24)*v21 + CV4(z1,24)*v22 + CV4(t1,24)*v23;
                    float32x4_t r15= vbase + CV4(x1,28)*v20 + CV4(y1,28)*v21 + CV4(z1,28)*v22 + CV4(t1,28)*v23;
                    *(float32x4_t*)(crow+j+0)+=r0; *(float32x4_t*)(crow+j+4)+=r1; *(float32x4_t*)(crow+j+8)+=r2; *(float32x4_t*)(crow+j+12)+=r3;
                    *(float32x4_t*)(crow+j+16)+=r4; *(float32x4_t*)(crow+j+20)+=r5; *(float32x4_t*)(crow+j+24)+=r6; *(float32x4_t*)(crow+j+28)+=r7;
                    *(float32x4_t*)(crow+j+32)+=r8; *(float32x4_t*)(crow+j+36)+=r9; *(float32x4_t*)(crow+j+40)+=r10; *(float32x4_t*)(crow+j+44)+=r11;
                    *(float32x4_t*)(crow+j+48)+=r12; *(float32x4_t*)(crow+j+52)+=r13; *(float32x4_t*)(crow+j+56)+=r14; *(float32x4_t*)(crow+j+60)+=r15;
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    uint32_t x=b0[w],y=b1[w],z=b2[w],t=b3[w];
                    *(float32x4_t*)(crow+j+0)+=vbase+CV4(x,0)*v20+CV4(y,0)*v21+CV4(z,0)*v22+CV4(t,0)*v23;
                    *(float32x4_t*)(crow+j+4)+=vbase+CV4(x,4)*v20+CV4(y,4)*v21+CV4(z,4)*v22+CV4(t,4)*v23;
                    *(float32x4_t*)(crow+j+8)+=vbase+CV4(x,8)*v20+CV4(y,8)*v21+CV4(z,8)*v22+CV4(t,8)*v23;
                    *(float32x4_t*)(crow+j+12)+=vbase+CV4(x,12)*v20+CV4(y,12)*v21+CV4(z,12)*v22+CV4(t,12)*v23;
                    *(float32x4_t*)(crow+j+16)+=vbase+CV4(x,16)*v20+CV4(y,16)*v21+CV4(z,16)*v22+CV4(t,16)*v23;
                    *(float32x4_t*)(crow+j+20)+=vbase+CV4(x,20)*v20+CV4(y,20)*v21+CV4(z,20)*v22+CV4(t,20)*v23;
                    *(float32x4_t*)(crow+j+24)+=vbase+CV4(x,24)*v20+CV4(y,24)*v21+CV4(z,24)*v22+CV4(t,24)*v23;
                    *(float32x4_t*)(crow+j+28)+=vbase+CV4(x,28)*v20+CV4(y,28)*v21+CV4(z,28)*v22+CV4(t,28)*v23;
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
                    uint32_t q0=brow[w], q1=brow[w+1];
                    float32x4_t r0=vna+CV4(q0,0)*vtwice, r1=vna+CV4(q0,4)*vtwice, r2=vna+CV4(q0,8)*vtwice, r3=vna+CV4(q0,12)*vtwice;
                    float32x4_t r4=vna+CV4(q0,16)*vtwice, r5=vna+CV4(q0,20)*vtwice, r6=vna+CV4(q0,24)*vtwice, r7=vna+CV4(q0,28)*vtwice;
                    float32x4_t r8=vna+CV4(q1,0)*vtwice, r9=vna+CV4(q1,4)*vtwice, r10=vna+CV4(q1,8)*vtwice, r11=vna+CV4(q1,12)*vtwice;
                    float32x4_t r12=vna+CV4(q1,16)*vtwice, r13=vna+CV4(q1,20)*vtwice, r14=vna+CV4(q1,24)*vtwice, r15=vna+CV4(q1,28)*vtwice;
                    *(float32x4_t*)(crow+j+0)=r0; *(float32x4_t*)(crow+j+4)=r1; *(float32x4_t*)(crow+j+8)=r2; *(float32x4_t*)(crow+j+12)=r3;
                    *(float32x4_t*)(crow+j+16)=r4; *(float32x4_t*)(crow+j+20)=r5; *(float32x4_t*)(crow+j+24)=r6; *(float32x4_t*)(crow+j+28)=r7;
                    *(float32x4_t*)(crow+j+32)=r8; *(float32x4_t*)(crow+j+36)=r9; *(float32x4_t*)(crow+j+40)=r10; *(float32x4_t*)(crow+j+44)=r11;
                    *(float32x4_t*)(crow+j+48)=r12; *(float32x4_t*)(crow+j+52)=r13; *(float32x4_t*)(crow+j+56)=r14; *(float32x4_t*)(crow+j+60)=r15;
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    uint32_t q=brow[w];
                    *(float32x4_t*)(crow+j+0)=vna+CV4(q,0)*vtwice; *(float32x4_t*)(crow+j+4)=vna+CV4(q,4)*vtwice;
                    *(float32x4_t*)(crow+j+8)=vna+CV4(q,8)*vtwice; *(float32x4_t*)(crow+j+12)=vna+CV4(q,12)*vtwice;
                    *(float32x4_t*)(crow+j+16)=vna+CV4(q,16)*vtwice; *(float32x4_t*)(crow+j+20)=vna+CV4(q,20)*vtwice;
                    *(float32x4_t*)(crow+j+24)=vna+CV4(q,24)*vtwice; *(float32x4_t*)(crow+j+28)=vna+CV4(q,28)*vtwice;
                }
            } else {
                for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                    uint32_t q0=brow[w], q1=brow[w+1];
                    float32x4_t r0=vna+CV4(q0,0)*vtwice, r1=vna+CV4(q0,4)*vtwice, r2=vna+CV4(q0,8)*vtwice, r3=vna+CV4(q0,12)*vtwice;
                    float32x4_t r4=vna+CV4(q0,16)*vtwice, r5=vna+CV4(q0,20)*vtwice, r6=vna+CV4(q0,24)*vtwice, r7=vna+CV4(q0,28)*vtwice;
                    float32x4_t r8=vna+CV4(q1,0)*vtwice, r9=vna+CV4(q1,4)*vtwice, r10=vna+CV4(q1,8)*vtwice, r11=vna+CV4(q1,12)*vtwice;
                    float32x4_t r12=vna+CV4(q1,16)*vtwice, r13=vna+CV4(q1,20)*vtwice, r14=vna+CV4(q1,24)*vtwice, r15=vna+CV4(q1,28)*vtwice;
                    *(float32x4_t*)(crow+j+0)+=r0; *(float32x4_t*)(crow+j+4)+=r1; *(float32x4_t*)(crow+j+8)+=r2; *(float32x4_t*)(crow+j+12)+=r3;
                    *(float32x4_t*)(crow+j+16)+=r4; *(float32x4_t*)(crow+j+20)+=r5; *(float32x4_t*)(crow+j+24)+=r6; *(float32x4_t*)(crow+j+28)+=r7;
                    *(float32x4_t*)(crow+j+32)+=r8; *(float32x4_t*)(crow+j+36)+=r9; *(float32x4_t*)(crow+j+40)+=r10; *(float32x4_t*)(crow+j+44)+=r11;
                    *(float32x4_t*)(crow+j+48)+=r12; *(float32x4_t*)(crow+j+52)+=r13; *(float32x4_t*)(crow+j+56)+=r14; *(float32x4_t*)(crow+j+60)+=r15;
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    uint32_t q=brow[w];
                    *(float32x4_t*)(crow+j+0)+=vna+CV4(q,0)*vtwice; *(float32x4_t*)(crow+j+4)+=vna+CV4(q,4)*vtwice;
                    *(float32x4_t*)(crow+j+8)+=vna+CV4(q,8)*vtwice; *(float32x4_t*)(crow+j+12)+=vna+CV4(q,12)*vtwice;
                    *(float32x4_t*)(crow+j+16)+=vna+CV4(q,16)*vtwice; *(float32x4_t*)(crow+j+20)+=vna+CV4(q,20)*vtwice;
                    *(float32x4_t*)(crow+j+24)+=vna+CV4(q,24)*vtwice; *(float32x4_t*)(crow+j+28)+=vna+CV4(q,28)*vtwice;
                }
            }
        }
    }
}
