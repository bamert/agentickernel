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
                for (size_t w=0; w+1<K_ints; w+=2, j+=64) {
                    const uint32_t x0=b0[w],x1=b1[w],x2=b2[w],x3=b3[w],x4=b4[w];
                    const uint32_t y0=b0[w+1],y1=b1[w+1],y2=b2[w+1],y3=b3[w+1],y4=b4[w+1];
                    float32x4_t* c0=(float32x4_t*)(crow+j+0);  float32x4_t* c1=(float32x4_t*)(crow+j+4);
                    float32x4_t* c2=(float32x4_t*)(crow+j+8);  float32x4_t* c3=(float32x4_t*)(crow+j+12);
                    float32x4_t* c4=(float32x4_t*)(crow+j+16); float32x4_t* c5=(float32x4_t*)(crow+j+20);
                    float32x4_t* c6=(float32x4_t*)(crow+j+24); float32x4_t* c7=(float32x4_t*)(crow+j+28);
                    float32x4_t* c8=(float32x4_t*)(crow+j+32); float32x4_t* c9=(float32x4_t*)(crow+j+36);
                    float32x4_t* c10=(float32x4_t*)(crow+j+40); float32x4_t* c11=(float32x4_t*)(crow+j+44);
                    float32x4_t* c12=(float32x4_t*)(crow+j+48); float32x4_t* c13=(float32x4_t*)(crow+j+52);
                    float32x4_t* c14=(float32x4_t*)(crow+j+56); float32x4_t* c15=(float32x4_t*)(crow+j+60);
                    *c0=vb+CV4(x0,0)*v0+CV4(x1,0)*v1+CV4(x2,0)*v2+CV4(x3,0)*v3+CV4(x4,0)*v4;
                    *c1=vb+CV4(x0,4)*v0+CV4(x1,4)*v1+CV4(x2,4)*v2+CV4(x3,4)*v3+CV4(x4,4)*v4;
                    *c2=vb+CV4(x0,8)*v0+CV4(x1,8)*v1+CV4(x2,8)*v2+CV4(x3,8)*v3+CV4(x4,8)*v4;
                    *c3=vb+CV4(x0,12)*v0+CV4(x1,12)*v1+CV4(x2,12)*v2+CV4(x3,12)*v3+CV4(x4,12)*v4;
                    *c4=vb+CV4(x0,16)*v0+CV4(x1,16)*v1+CV4(x2,16)*v2+CV4(x3,16)*v3+CV4(x4,16)*v4;
                    *c5=vb+CV4(x0,20)*v0+CV4(x1,20)*v1+CV4(x2,20)*v2+CV4(x3,20)*v3+CV4(x4,20)*v4;
                    *c6=vb+CV4(x0,24)*v0+CV4(x1,24)*v1+CV4(x2,24)*v2+CV4(x3,24)*v3+CV4(x4,24)*v4;
                    *c7=vb+CV4(x0,28)*v0+CV4(x1,28)*v1+CV4(x2,28)*v2+CV4(x3,28)*v3+CV4(x4,28)*v4;
                    *c8=vb+CV4(y0,0)*v0+CV4(y1,0)*v1+CV4(y2,0)*v2+CV4(y3,0)*v3+CV4(y4,0)*v4;
                    *c9=vb+CV4(y0,4)*v0+CV4(y1,4)*v1+CV4(y2,4)*v2+CV4(y3,4)*v3+CV4(y4,4)*v4;
                    *c10=vb+CV4(y0,8)*v0+CV4(y1,8)*v1+CV4(y2,8)*v2+CV4(y3,8)*v3+CV4(y4,8)*v4;
                    *c11=vb+CV4(y0,12)*v0+CV4(y1,12)*v1+CV4(y2,12)*v2+CV4(y3,12)*v3+CV4(y4,12)*v4;
                    *c12=vb+CV4(y0,16)*v0+CV4(y1,16)*v1+CV4(y2,16)*v2+CV4(y3,16)*v3+CV4(y4,16)*v4;
                    *c13=vb+CV4(y0,20)*v0+CV4(y1,20)*v1+CV4(y2,20)*v2+CV4(y3,20)*v3+CV4(y4,20)*v4;
                    *c14=vb+CV4(y0,24)*v0+CV4(y1,24)*v1+CV4(y2,24)*v2+CV4(y3,24)*v3+CV4(y4,24)*v4;
                    *c15=vb+CV4(y0,28)*v0+CV4(y1,28)*v1+CV4(y2,28)*v2+CV4(y3,28)*v3+CV4(y4,28)*v4;
                }
                for (size_t w=(K_ints & ~size_t(1)); w<K_ints; ++w, j+=32) {
                    const uint32_t x0=b0[w],x1=b1[w],x2=b2[w],x3=b3[w],x4=b4[w];
                    *(float32x4_t*)(crow+j+0)=vb+CV4(x0,0)*v0+CV4(x1,0)*v1+CV4(x2,0)*v2+CV4(x3,0)*v3+CV4(x4,0)*v4;
                    *(float32x4_t*)(crow+j+4)=vb+CV4(x0,4)*v0+CV4(x1,4)*v1+CV4(x2,4)*v2+CV4(x3,4)*v3+CV4(x4,4)*v4;
                    *(float32x4_t*)(crow+j+8)=vb+CV4(x0,8)*v0+CV4(x1,8)*v1+CV4(x2,8)*v2+CV4(x3,8)*v3+CV4(x4,8)*v4;
                    *(float32x4_t*)(crow+j+12)=vb+CV4(x0,12)*v0+CV4(x1,12)*v1+CV4(x2,12)*v2+CV4(x3,12)*v3+CV4(x4,12)*v4;
                    *(float32x4_t*)(crow+j+16)=vb+CV4(x0,16)*v0+CV4(x1,16)*v1+CV4(x2,16)*v2+CV4(x3,16)*v3+CV4(x4,16)*v4;
                    *(float32x4_t*)(crow+j+20)=vb+CV4(x0,20)*v0+CV4(x1,20)*v1+CV4(x2,20)*v2+CV4(x3,20)*v3+CV4(x4,20)*v4;
                    *(float32x4_t*)(crow+j+24)=vb+CV4(x0,24)*v0+CV4(x1,24)*v1+CV4(x2,24)*v2+CV4(x3,24)*v3+CV4(x4,24)*v4;
                    *(float32x4_t*)(crow+j+28)=vb+CV4(x0,28)*v0+CV4(x1,28)*v1+CV4(x2,28)*v2+CV4(x3,28)*v3+CV4(x4,28)*v4;
                }
            } else {
                for (size_t w=0; w+1<K_ints; w+=2, j+=64) {
                    const uint32_t x0=b0[w],x1=b1[w],x2=b2[w],x3=b3[w],x4=b4[w];
                    const uint32_t y0=b0[w+1],y1=b1[w+1],y2=b2[w+1],y3=b3[w+1],y4=b4[w+1];
                    float32x4_t* c0=(float32x4_t*)(crow+j+0);  float32x4_t* c1=(float32x4_t*)(crow+j+4);
                    float32x4_t* c2=(float32x4_t*)(crow+j+8);  float32x4_t* c3=(float32x4_t*)(crow+j+12);
                    float32x4_t* c4=(float32x4_t*)(crow+j+16); float32x4_t* c5=(float32x4_t*)(crow+j+20);
                    float32x4_t* c6=(float32x4_t*)(crow+j+24); float32x4_t* c7=(float32x4_t*)(crow+j+28);
                    float32x4_t* c8=(float32x4_t*)(crow+j+32); float32x4_t* c9=(float32x4_t*)(crow+j+36);
                    float32x4_t* c10=(float32x4_t*)(crow+j+40); float32x4_t* c11=(float32x4_t*)(crow+j+44);
                    float32x4_t* c12=(float32x4_t*)(crow+j+48); float32x4_t* c13=(float32x4_t*)(crow+j+52);
                    float32x4_t* c14=(float32x4_t*)(crow+j+56); float32x4_t* c15=(float32x4_t*)(crow+j+60);
                    *c0+=vb+CV4(x0,0)*v0+CV4(x1,0)*v1+CV4(x2,0)*v2+CV4(x3,0)*v3+CV4(x4,0)*v4;
                    *c1+=vb+CV4(x0,4)*v0+CV4(x1,4)*v1+CV4(x2,4)*v2+CV4(x3,4)*v3+CV4(x4,4)*v4;
                    *c2+=vb+CV4(x0,8)*v0+CV4(x1,8)*v1+CV4(x2,8)*v2+CV4(x3,8)*v3+CV4(x4,8)*v4;
                    *c3+=vb+CV4(x0,12)*v0+CV4(x1,12)*v1+CV4(x2,12)*v2+CV4(x3,12)*v3+CV4(x4,12)*v4;
                    *c4+=vb+CV4(x0,16)*v0+CV4(x1,16)*v1+CV4(x2,16)*v2+CV4(x3,16)*v3+CV4(x4,16)*v4;
                    *c5+=vb+CV4(x0,20)*v0+CV4(x1,20)*v1+CV4(x2,20)*v2+CV4(x3,20)*v3+CV4(x4,20)*v4;
                    *c6+=vb+CV4(x0,24)*v0+CV4(x1,24)*v1+CV4(x2,24)*v2+CV4(x3,24)*v3+CV4(x4,24)*v4;
                    *c7+=vb+CV4(x0,28)*v0+CV4(x1,28)*v1+CV4(x2,28)*v2+CV4(x3,28)*v3+CV4(x4,28)*v4;
                    *c8+=vb+CV4(y0,0)*v0+CV4(y1,0)*v1+CV4(y2,0)*v2+CV4(y3,0)*v3+CV4(y4,0)*v4;
                    *c9+=vb+CV4(y0,4)*v0+CV4(y1,4)*v1+CV4(y2,4)*v2+CV4(y3,4)*v3+CV4(y4,4)*v4;
                    *c10+=vb+CV4(y0,8)*v0+CV4(y1,8)*v1+CV4(y2,8)*v2+CV4(y3,8)*v3+CV4(y4,8)*v4;
                    *c11+=vb+CV4(y0,12)*v0+CV4(y1,12)*v1+CV4(y2,12)*v2+CV4(y3,12)*v3+CV4(y4,12)*v4;
                    *c12+=vb+CV4(y0,16)*v0+CV4(y1,16)*v1+CV4(y2,16)*v2+CV4(y3,16)*v3+CV4(y4,16)*v4;
                    *c13+=vb+CV4(y0,20)*v0+CV4(y1,20)*v1+CV4(y2,20)*v2+CV4(y3,20)*v3+CV4(y4,20)*v4;
                    *c14+=vb+CV4(y0,24)*v0+CV4(y1,24)*v1+CV4(y2,24)*v2+CV4(y3,24)*v3+CV4(y4,24)*v4;
                    *c15+=vb+CV4(y0,28)*v0+CV4(y1,28)*v1+CV4(y2,28)*v2+CV4(y3,28)*v3+CV4(y4,28)*v4;
                }
                for (size_t w=(K_ints & ~size_t(1)); w<K_ints; ++w, j+=32) {
                    const uint32_t x0=b0[w],x1=b1[w],x2=b2[w],x3=b3[w],x4=b4[w];
                    *(float32x4_t*)(crow+j+0)+=vb+CV4(x0,0)*v0+CV4(x1,0)*v1+CV4(x2,0)*v2+CV4(x3,0)*v3+CV4(x4,0)*v4;
                    *(float32x4_t*)(crow+j+4)+=vb+CV4(x0,4)*v0+CV4(x1,4)*v1+CV4(x2,4)*v2+CV4(x3,4)*v3+CV4(x4,4)*v4;
                    *(float32x4_t*)(crow+j+8)+=vb+CV4(x0,8)*v0+CV4(x1,8)*v1+CV4(x2,8)*v2+CV4(x3,8)*v3+CV4(x4,8)*v4;
                    *(float32x4_t*)(crow+j+12)+=vb+CV4(x0,12)*v0+CV4(x1,12)*v1+CV4(x2,12)*v2+CV4(x3,12)*v3+CV4(x4,12)*v4;
                    *(float32x4_t*)(crow+j+16)+=vb+CV4(x0,16)*v0+CV4(x1,16)*v1+CV4(x2,16)*v2+CV4(x3,16)*v3+CV4(x4,16)*v4;
                    *(float32x4_t*)(crow+j+20)+=vb+CV4(x0,20)*v0+CV4(x1,20)*v1+CV4(x2,20)*v2+CV4(x3,20)*v3+CV4(x4,20)*v4;
                    *(float32x4_t*)(crow+j+24)+=vb+CV4(x0,24)*v0+CV4(x1,24)*v1+CV4(x2,24)*v2+CV4(x3,24)*v3+CV4(x4,24)*v4;
                    *(float32x4_t*)(crow+j+28)+=vb+CV4(x0,28)*v0+CV4(x1,28)*v1+CV4(x2,28)*v2+CV4(x3,28)*v3+CV4(x4,28)*v4;
                }
            }
        }
        for (size_t p=(K/5)*5; p<K; ++p) {
            const float a=arow[p];
            const float32x4_t vna={-a,-a,-a,-a}, vtw={a+a,a+a,a+a,a+a};
            const uint32_t* brow=B+p*K_ints;
            size_t j=0;
            if (p==0) {
                for (size_t w=0; w+1<K_ints; w+=2, j+=64) {
                    const uint32_t q0=brow[w], q1=brow[w+1];
                    *(float32x4_t*)(crow+j+0)=vna+CV4(q0,0)*vtw; *(float32x4_t*)(crow+j+4)=vna+CV4(q0,4)*vtw;
                    *(float32x4_t*)(crow+j+8)=vna+CV4(q0,8)*vtw; *(float32x4_t*)(crow+j+12)=vna+CV4(q0,12)*vtw;
                    *(float32x4_t*)(crow+j+16)=vna+CV4(q0,16)*vtw; *(float32x4_t*)(crow+j+20)=vna+CV4(q0,20)*vtw;
                    *(float32x4_t*)(crow+j+24)=vna+CV4(q0,24)*vtw; *(float32x4_t*)(crow+j+28)=vna+CV4(q0,28)*vtw;
                    *(float32x4_t*)(crow+j+32)=vna+CV4(q1,0)*vtw; *(float32x4_t*)(crow+j+36)=vna+CV4(q1,4)*vtw;
                    *(float32x4_t*)(crow+j+40)=vna+CV4(q1,8)*vtw; *(float32x4_t*)(crow+j+44)=vna+CV4(q1,12)*vtw;
                    *(float32x4_t*)(crow+j+48)=vna+CV4(q1,16)*vtw; *(float32x4_t*)(crow+j+52)=vna+CV4(q1,20)*vtw;
                    *(float32x4_t*)(crow+j+56)=vna+CV4(q1,24)*vtw; *(float32x4_t*)(crow+j+60)=vna+CV4(q1,28)*vtw;
                }
                for (size_t w=(K_ints & ~size_t(1)); w<K_ints; ++w, j+=32) {
                    const uint32_t q=brow[w];
                    *(float32x4_t*)(crow+j+0)=vna+CV4(q,0)*vtw; *(float32x4_t*)(crow+j+4)=vna+CV4(q,4)*vtw;
                    *(float32x4_t*)(crow+j+8)=vna+CV4(q,8)*vtw; *(float32x4_t*)(crow+j+12)=vna+CV4(q,12)*vtw;
                    *(float32x4_t*)(crow+j+16)=vna+CV4(q,16)*vtw; *(float32x4_t*)(crow+j+20)=vna+CV4(q,20)*vtw;
                    *(float32x4_t*)(crow+j+24)=vna+CV4(q,24)*vtw; *(float32x4_t*)(crow+j+28)=vna+CV4(q,28)*vtw;
                }
            } else {
                for (size_t w=0; w+1<K_ints; w+=2, j+=64) {
                    const uint32_t q0=brow[w], q1=brow[w+1];
                    *(float32x4_t*)(crow+j+0)+=vna+CV4(q0,0)*vtw; *(float32x4_t*)(crow+j+4)+=vna+CV4(q0,4)*vtw;
                    *(float32x4_t*)(crow+j+8)+=vna+CV4(q0,8)*vtw; *(float32x4_t*)(crow+j+12)+=vna+CV4(q0,12)*vtw;
                    *(float32x4_t*)(crow+j+16)+=vna+CV4(q0,16)*vtw; *(float32x4_t*)(crow+j+20)+=vna+CV4(q0,20)*vtw;
                    *(float32x4_t*)(crow+j+24)+=vna+CV4(q0,24)*vtw; *(float32x4_t*)(crow+j+28)+=vna+CV4(q0,28)*vtw;
                    *(float32x4_t*)(crow+j+32)+=vna+CV4(q1,0)*vtw; *(float32x4_t*)(crow+j+36)+=vna+CV4(q1,4)*vtw;
                    *(float32x4_t*)(crow+j+40)+=vna+CV4(q1,8)*vtw; *(float32x4_t*)(crow+j+44)+=vna+CV4(q1,12)*vtw;
                    *(float32x4_t*)(crow+j+48)+=vna+CV4(q1,16)*vtw; *(float32x4_t*)(crow+j+52)+=vna+CV4(q1,20)*vtw;
                    *(float32x4_t*)(crow+j+56)+=vna+CV4(q1,24)*vtw; *(float32x4_t*)(crow+j+60)+=vna+CV4(q1,28)*vtw;
                }
                for (size_t w=(K_ints & ~size_t(1)); w<K_ints; ++w, j+=32) {
                    const uint32_t q=brow[w];
                    *(float32x4_t*)(crow+j+0)+=vna+CV4(q,0)*vtw; *(float32x4_t*)(crow+j+4)+=vna+CV4(q,4)*vtw;
                    *(float32x4_t*)(crow+j+8)+=vna+CV4(q,8)*vtw; *(float32x4_t*)(crow+j+12)+=vna+CV4(q,12)*vtw;
                    *(float32x4_t*)(crow+j+16)+=vna+CV4(q,16)*vtw; *(float32x4_t*)(crow+j+20)+=vna+CV4(q,20)*vtw;
                    *(float32x4_t*)(crow+j+24)+=vna+CV4(q,24)*vtw; *(float32x4_t*)(crow+j+28)+=vna+CV4(q,28)*vtw;
                }
            }
        }
    }
}
