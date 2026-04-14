#pragma once

typedef unsigned int uint32_t;
typedef decltype(sizeof(0)) size_t;

typedef float float32x4_t __attribute__((vector_size(16)));
typedef unsigned int uint32x4_t __attribute__((vector_size(16)));

#define V4(bits, sh) __builtin_convertvector((uint32x4_t){((bits)>>((sh)+0))&1u,((bits)>>((sh)+1))&1u,((bits)>>((sh)+2))&1u,((bits)>>((sh)+3))&1u}, float32x4_t)
#define DO16_SET(baseptr, x0,y0,z0,t0, x1,y1,z1,t1, vbase,v20,v21,v22,v23) \
    *(float32x4_t*)((baseptr)+0)  = (vbase) + V4(x0,0)*(v20)  + V4(y0,0)*(v21)  + V4(z0,0)*(v22)  + V4(t0,0)*(v23); \
    *(float32x4_t*)((baseptr)+4)  = (vbase) + V4(x0,4)*(v20)  + V4(y0,4)*(v21)  + V4(z0,4)*(v22)  + V4(t0,4)*(v23); \
    *(float32x4_t*)((baseptr)+8)  = (vbase) + V4(x0,8)*(v20)  + V4(y0,8)*(v21)  + V4(z0,8)*(v22)  + V4(t0,8)*(v23); \
    *(float32x4_t*)((baseptr)+12) = (vbase) + V4(x0,12)*(v20) + V4(y0,12)*(v21) + V4(z0,12)*(v22) + V4(t0,12)*(v23); \
    *(float32x4_t*)((baseptr)+16) = (vbase) + V4(x0,16)*(v20) + V4(y0,16)*(v21) + V4(z0,16)*(v22) + V4(t0,16)*(v23); \
    *(float32x4_t*)((baseptr)+20) = (vbase) + V4(x0,20)*(v20) + V4(y0,20)*(v21) + V4(z0,20)*(v22) + V4(t0,20)*(v23); \
    *(float32x4_t*)((baseptr)+24) = (vbase) + V4(x0,24)*(v20) + V4(y0,24)*(v21) + V4(z0,24)*(v22) + V4(t0,24)*(v23); \
    *(float32x4_t*)((baseptr)+28) = (vbase) + V4(x0,28)*(v20) + V4(y0,28)*(v21) + V4(z0,28)*(v22) + V4(t0,28)*(v23); \
    *(float32x4_t*)((baseptr)+32) = (vbase) + V4(x1,0)*(v20)  + V4(y1,0)*(v21)  + V4(z1,0)*(v22)  + V4(t1,0)*(v23); \
    *(float32x4_t*)((baseptr)+36) = (vbase) + V4(x1,4)*(v20)  + V4(y1,4)*(v21)  + V4(z1,4)*(v22)  + V4(t1,4)*(v23); \
    *(float32x4_t*)((baseptr)+40) = (vbase) + V4(x1,8)*(v20)  + V4(y1,8)*(v21)  + V4(z1,8)*(v22)  + V4(t1,8)*(v23); \
    *(float32x4_t*)((baseptr)+44) = (vbase) + V4(x1,12)*(v20) + V4(y1,12)*(v21) + V4(z1,12)*(v22) + V4(t1,12)*(v23); \
    *(float32x4_t*)((baseptr)+48) = (vbase) + V4(x1,16)*(v20) + V4(y1,16)*(v21) + V4(z1,16)*(v22) + V4(t1,16)*(v23); \
    *(float32x4_t*)((baseptr)+52) = (vbase) + V4(x1,20)*(v20) + V4(y1,20)*(v21) + V4(z1,20)*(v22) + V4(t1,20)*(v23); \
    *(float32x4_t*)((baseptr)+56) = (vbase) + V4(x1,24)*(v20) + V4(y1,24)*(v21) + V4(z1,24)*(v22) + V4(t1,24)*(v23); \
    *(float32x4_t*)((baseptr)+60) = (vbase) + V4(x1,28)*(v20) + V4(y1,28)*(v21) + V4(z1,28)*(v22) + V4(t1,28)*(v23)
#define DO16_ADD(baseptr, x0,y0,z0,t0, x1,y1,z1,t1, vbase,v20,v21,v22,v23) \
    *(float32x4_t*)((baseptr)+0)  += (vbase) + V4(x0,0)*(v20)  + V4(y0,0)*(v21)  + V4(z0,0)*(v22)  + V4(t0,0)*(v23); \
    *(float32x4_t*)((baseptr)+4)  += (vbase) + V4(x0,4)*(v20)  + V4(y0,4)*(v21)  + V4(z0,4)*(v22)  + V4(t0,4)*(v23); \
    *(float32x4_t*)((baseptr)+8)  += (vbase) + V4(x0,8)*(v20)  + V4(y0,8)*(v21)  + V4(z0,8)*(v22)  + V4(t0,8)*(v23); \
    *(float32x4_t*)((baseptr)+12) += (vbase) + V4(x0,12)*(v20) + V4(y0,12)*(v21) + V4(z0,12)*(v22) + V4(t0,12)*(v23); \
    *(float32x4_t*)((baseptr)+16) += (vbase) + V4(x0,16)*(v20) + V4(y0,16)*(v21) + V4(z0,16)*(v22) + V4(t0,16)*(v23); \
    *(float32x4_t*)((baseptr)+20) += (vbase) + V4(x0,20)*(v20) + V4(y0,20)*(v21) + V4(z0,20)*(v22) + V4(t0,20)*(v23); \
    *(float32x4_t*)((baseptr)+24) += (vbase) + V4(x0,24)*(v20) + V4(y0,24)*(v21) + V4(z0,24)*(v22) + V4(t0,24)*(v23); \
    *(float32x4_t*)((baseptr)+28) += (vbase) + V4(x0,28)*(v20) + V4(y0,28)*(v21) + V4(z0,28)*(v22) + V4(t0,28)*(v23); \
    *(float32x4_t*)((baseptr)+32) += (vbase) + V4(x1,0)*(v20)  + V4(y1,0)*(v21)  + V4(z1,0)*(v22)  + V4(t1,0)*(v23); \
    *(float32x4_t*)((baseptr)+36) += (vbase) + V4(x1,4)*(v20)  + V4(y1,4)*(v21)  + V4(z1,4)*(v22)  + V4(t1,4)*(v23); \
    *(float32x4_t*)((baseptr)+40) += (vbase) + V4(x1,8)*(v20)  + V4(y1,8)*(v21)  + V4(z1,8)*(v22)  + V4(t1,8)*(v23); \
    *(float32x4_t*)((baseptr)+44) += (vbase) + V4(x1,12)*(v20) + V4(y1,12)*(v21) + V4(z1,12)*(v22) + V4(t1,12)*(v23); \
    *(float32x4_t*)((baseptr)+48) += (vbase) + V4(x1,16)*(v20) + V4(y1,16)*(v21) + V4(z1,16)*(v22) + V4(t1,16)*(v23); \
    *(float32x4_t*)((baseptr)+52) += (vbase) + V4(x1,20)*(v20) + V4(y1,20)*(v21) + V4(z1,20)*(v22) + V4(t1,20)*(v23); \
    *(float32x4_t*)((baseptr)+56) += (vbase) + V4(x1,24)*(v20) + V4(y1,24)*(v21) + V4(z1,24)*(v22) + V4(t1,24)*(v23); \
    *(float32x4_t*)((baseptr)+60) += (vbase) + V4(x1,28)*(v20) + V4(y1,28)*(v21) + V4(z1,28)*(v22) + V4(t1,28)*(v23)

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
                    DO16_SET(crow + j, b0[w],b1[w],b2[w],b3[w], b0[w+1],b1[w+1],b2[w+1],b3[w+1], vbase,v20,v21,v22,v23);
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    *(float32x4_t*)(crow + j + 0) = vbase + V4(b0[w],0)*v20 + V4(b1[w],0)*v21 + V4(b2[w],0)*v22 + V4(b3[w],0)*v23;
                    *(float32x4_t*)(crow + j + 4) = vbase + V4(b0[w],4)*v20 + V4(b1[w],4)*v21 + V4(b2[w],4)*v22 + V4(b3[w],4)*v23;
                    *(float32x4_t*)(crow + j + 8) = vbase + V4(b0[w],8)*v20 + V4(b1[w],8)*v21 + V4(b2[w],8)*v22 + V4(b3[w],8)*v23;
                    *(float32x4_t*)(crow + j + 12)= vbase + V4(b0[w],12)*v20+ V4(b1[w],12)*v21+ V4(b2[w],12)*v22+ V4(b3[w],12)*v23;
                    *(float32x4_t*)(crow + j + 16)= vbase + V4(b0[w],16)*v20+ V4(b1[w],16)*v21+ V4(b2[w],16)*v22+ V4(b3[w],16)*v23;
                    *(float32x4_t*)(crow + j + 20)= vbase + V4(b0[w],20)*v20+ V4(b1[w],20)*v21+ V4(b2[w],20)*v22+ V4(b3[w],20)*v23;
                    *(float32x4_t*)(crow + j + 24)= vbase + V4(b0[w],24)*v20+ V4(b1[w],24)*v21+ V4(b2[w],24)*v22+ V4(b3[w],24)*v23;
                    *(float32x4_t*)(crow + j + 28)= vbase + V4(b0[w],28)*v20+ V4(b1[w],28)*v21+ V4(b2[w],28)*v22+ V4(b3[w],28)*v23;
                }
            } else {
                for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                    DO16_ADD(crow + j, b0[w],b1[w],b2[w],b3[w], b0[w+1],b1[w+1],b2[w+1],b3[w+1], vbase,v20,v21,v22,v23);
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    *(float32x4_t*)(crow + j + 0) += vbase + V4(b0[w],0)*v20 + V4(b1[w],0)*v21 + V4(b2[w],0)*v22 + V4(b3[w],0)*v23;
                    *(float32x4_t*)(crow + j + 4) += vbase + V4(b0[w],4)*v20 + V4(b1[w],4)*v21 + V4(b2[w],4)*v22 + V4(b3[w],4)*v23;
                    *(float32x4_t*)(crow + j + 8) += vbase + V4(b0[w],8)*v20 + V4(b1[w],8)*v21 + V4(b2[w],8)*v22 + V4(b3[w],8)*v23;
                    *(float32x4_t*)(crow + j + 12)+= vbase + V4(b0[w],12)*v20+ V4(b1[w],12)*v21+ V4(b2[w],12)*v22+ V4(b3[w],12)*v23;
                    *(float32x4_t*)(crow + j + 16)+= vbase + V4(b0[w],16)*v20+ V4(b1[w],16)*v21+ V4(b2[w],16)*v22+ V4(b3[w],16)*v23;
                    *(float32x4_t*)(crow + j + 20)+= vbase + V4(b0[w],20)*v20+ V4(b1[w],20)*v21+ V4(b2[w],20)*v22+ V4(b3[w],20)*v23;
                    *(float32x4_t*)(crow + j + 24)+= vbase + V4(b0[w],24)*v20+ V4(b1[w],24)*v21+ V4(b2[w],24)*v22+ V4(b3[w],24)*v23;
                    *(float32x4_t*)(crow + j + 28)+= vbase + V4(b0[w],28)*v20+ V4(b1[w],28)*v21+ V4(b2[w],28)*v22+ V4(b3[w],28)*v23;
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
                    *(float32x4_t*)(crow + j + 0)  = vna + V4(q0,0)*vtwice; *(float32x4_t*)(crow + j + 4)  = vna + V4(q0,4)*vtwice;
                    *(float32x4_t*)(crow + j + 8)  = vna + V4(q0,8)*vtwice; *(float32x4_t*)(crow + j + 12) = vna + V4(q0,12)*vtwice;
                    *(float32x4_t*)(crow + j + 16) = vna + V4(q0,16)*vtwice;*(float32x4_t*)(crow + j + 20) = vna + V4(q0,20)*vtwice;
                    *(float32x4_t*)(crow + j + 24) = vna + V4(q0,24)*vtwice;*(float32x4_t*)(crow + j + 28) = vna + V4(q0,28)*vtwice;
                    *(float32x4_t*)(crow + j + 32) = vna + V4(q1,0)*vtwice; *(float32x4_t*)(crow + j + 36) = vna + V4(q1,4)*vtwice;
                    *(float32x4_t*)(crow + j + 40) = vna + V4(q1,8)*vtwice; *(float32x4_t*)(crow + j + 44) = vna + V4(q1,12)*vtwice;
                    *(float32x4_t*)(crow + j + 48) = vna + V4(q1,16)*vtwice;*(float32x4_t*)(crow + j + 52) = vna + V4(q1,20)*vtwice;
                    *(float32x4_t*)(crow + j + 56) = vna + V4(q1,24)*vtwice;*(float32x4_t*)(crow + j + 60) = vna + V4(q1,28)*vtwice;
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    uint32_t q=brow[w];
                    *(float32x4_t*)(crow + j + 0)=vna+V4(q,0)*vtwice; *(float32x4_t*)(crow + j + 4)=vna+V4(q,4)*vtwice;
                    *(float32x4_t*)(crow + j + 8)=vna+V4(q,8)*vtwice; *(float32x4_t*)(crow + j + 12)=vna+V4(q,12)*vtwice;
                    *(float32x4_t*)(crow + j + 16)=vna+V4(q,16)*vtwice;*(float32x4_t*)(crow + j + 20)=vna+V4(q,20)*vtwice;
                    *(float32x4_t*)(crow + j + 24)=vna+V4(q,24)*vtwice;*(float32x4_t*)(crow + j + 28)=vna+V4(q,28)*vtwice;
                }
            } else {
                for (size_t w = 0; w + 1 < K_ints; w += 2, j += 64) {
                    uint32_t q0=brow[w], q1=brow[w+1];
                    *(float32x4_t*)(crow + j + 0)  += vna + V4(q0,0)*vtwice; *(float32x4_t*)(crow + j + 4)  += vna + V4(q0,4)*vtwice;
                    *(float32x4_t*)(crow + j + 8)  += vna + V4(q0,8)*vtwice; *(float32x4_t*)(crow + j + 12) += vna + V4(q0,12)*vtwice;
                    *(float32x4_t*)(crow + j + 16) += vna + V4(q0,16)*vtwice;*(float32x4_t*)(crow + j + 20) += vna + V4(q0,20)*vtwice;
                    *(float32x4_t*)(crow + j + 24) += vna + V4(q0,24)*vtwice;*(float32x4_t*)(crow + j + 28) += vna + V4(q0,28)*vtwice;
                    *(float32x4_t*)(crow + j + 32) += vna + V4(q1,0)*vtwice; *(float32x4_t*)(crow + j + 36) += vna + V4(q1,4)*vtwice;
                    *(float32x4_t*)(crow + j + 40) += vna + V4(q1,8)*vtwice; *(float32x4_t*)(crow + j + 44) += vna + V4(q1,12)*vtwice;
                    *(float32x4_t*)(crow + j + 48) += vna + V4(q1,16)*vtwice;*(float32x4_t*)(crow + j + 52) += vna + V4(q1,20)*vtwice;
                    *(float32x4_t*)(crow + j + 56) += vna + V4(q1,24)*vtwice;*(float32x4_t*)(crow + j + 60) += vna + V4(q1,28)*vtwice;
                }
                for (size_t w = (K_ints & ~size_t(1)); w < K_ints; ++w, j += 32) {
                    uint32_t q=brow[w];
                    *(float32x4_t*)(crow + j + 0)+=vna+V4(q,0)*vtwice; *(float32x4_t*)(crow + j + 4)+=vna+V4(q,4)*vtwice;
                    *(float32x4_t*)(crow + j + 8)+=vna+V4(q,8)*vtwice; *(float32x4_t*)(crow + j + 12)+=vna+V4(q,12)*vtwice;
                    *(float32x4_t*)(crow + j + 16)+=vna+V4(q,16)*vtwice;*(float32x4_t*)(crow + j + 20)+=vna+V4(q,20)*vtwice;
                    *(float32x4_t*)(crow + j + 24)+=vna+V4(q,24)*vtwice;*(float32x4_t*)(crow + j + 28)+=vna+V4(q,28)*vtwice;
                }
            }
        }
    }
}
