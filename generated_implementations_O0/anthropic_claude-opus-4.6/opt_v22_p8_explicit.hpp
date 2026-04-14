
#pragma once
#include <cstdint>
#include <cstddef>
#include <string.h>

// v19 but with fully explicit unrolled p=8, no arrays for a values and B pointers
void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, 
            float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    size_t i = 0;
    for (; i + 4 <= M; i += 4) {
        float* __restrict__ C0 = C + (i + 0) * K;
        float* __restrict__ C1 = C + (i + 1) * K;
        float* __restrict__ C2 = C + (i + 2) * K;
        float* __restrict__ C3 = C + (i + 3) * K;
        const float* __restrict__ A0 = A + (i + 0) * K;
        const float* __restrict__ A1 = A + (i + 1) * K;
        const float* __restrict__ A2 = A + (i + 2) * K;
        const float* __restrict__ A3 = A + (i + 3) * K;

        memset(C0, 0, K * sizeof(float));
        memset(C1, 0, K * sizeof(float));
        memset(C2, 0, K * sizeof(float));
        memset(C3, 0, K * sizeof(float));

        size_t p = 0;
        for (; p + 8 <= K; p += 8) {
            float a00=A0[p],a01=A0[p+1],a02=A0[p+2],a03=A0[p+3],a04=A0[p+4],a05=A0[p+5],a06=A0[p+6],a07=A0[p+7];
            float a10=A1[p],a11=A1[p+1],a12=A1[p+2],a13=A1[p+3],a14=A1[p+4],a15=A1[p+5],a16=A1[p+6],a17=A1[p+7];
            float a20=A2[p],a21=A2[p+1],a22=A2[p+2],a23=A2[p+3],a24=A2[p+4],a25=A2[p+5],a26=A2[p+6],a27=A2[p+7];
            float a30=A3[p],a31=A3[p+1],a32=A3[p+2],a33=A3[p+3],a34=A3[p+4],a35=A3[p+5],a36=A3[p+6],a37=A3[p+7];
            
            const uint32_t* Bp0=B+(p+0)*K_ints, *Bp1=B+(p+1)*K_ints, *Bp2=B+(p+2)*K_ints, *Bp3=B+(p+3)*K_ints;
            const uint32_t* Bp4=B+(p+4)*K_ints, *Bp5=B+(p+5)*K_ints, *Bp6=B+(p+6)*K_ints, *Bp7=B+(p+7)*K_ints;

            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t pk0=Bp0[g],pk1=Bp1[g],pk2=Bp2[g],pk3=Bp3[g];
                uint32_t pk4=Bp4[g],pk5=Bp5[g],pk6=Bp6[g],pk7=Bp7[g];
                
                float* c0=C0+g*32, *c1=C1+g*32, *c2=C2+g*32, *c3=C3+g*32;

                for (int b = 0; b < 32; ++b) {
                    uint32_t mask = 1u << b;
                    float s0 = (pk0&mask)?1.0f:-1.0f;
                    float s1 = (pk1&mask)?1.0f:-1.0f;
                    float s2 = (pk2&mask)?1.0f:-1.0f;
                    float s3 = (pk3&mask)?1.0f:-1.0f;
                    float s4 = (pk4&mask)?1.0f:-1.0f;
                    float s5 = (pk5&mask)?1.0f:-1.0f;
                    float s6 = (pk6&mask)?1.0f:-1.0f;
                    float s7 = (pk7&mask)?1.0f:-1.0f;
                    
                    c0[b] += a00*s0+a01*s1+a02*s2+a03*s3+a04*s4+a05*s5+a06*s6+a07*s7;
                    c1[b] += a10*s0+a11*s1+a12*s2+a13*s3+a14*s4+a15*s5+a16*s6+a17*s7;
                    c2[b] += a20*s0+a21*s1+a22*s2+a23*s3+a24*s4+a25*s5+a26*s6+a27*s7;
                    c3[b] += a30*s0+a31*s1+a32*s2+a33*s3+a34*s4+a35*s5+a36*s6+a37*s7;
                }
            }
        }
        for (; p < K; ++p) {
            float a0v=A0[p],a1v=A1[p],a2v=A2[p],a3v=A3[p];
            const uint32_t* Bp = B+p*K_ints;
            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t pk = Bp[g];
                float* c0=C0+g*32, *c1=C1+g*32, *c2=C2+g*32, *c3=C3+g*32;
                for (int b = 0; b < 32; ++b) {
                    float s = (pk&(1u<<b))?1.0f:-1.0f;
                    c0[b]+=a0v*s; c1[b]+=a1v*s; c2[b]+=a2v*s; c3[b]+=a3v*s;
                }
            }
        }
    }

    for (; i < M; ++i) {
        float* Cr = C+i*K;
        const float* Ar = A+i*K;
        memset(Cr, 0, K*sizeof(float));
        size_t p = 0;
        for (; p + 8 <= K; p += 8) {
            float av[8];
            const uint32_t* Bps[8];
            for (int q=0;q<8;++q) { av[q]=Ar[p+q]; Bps[q]=B+(p+q)*K_ints; }
            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t pk[8];
                for (int q=0;q<8;++q) pk[q]=Bps[q][g];
                float* co = Cr+g*32;
                for (int b = 0; b < 32; ++b) {
                    uint32_t mask = 1u<<b;
                    float v = 0;
                    for (int q=0;q<8;++q) v += av[q]*((pk[q]&mask)?1.0f:-1.0f);
                    co[b] += v;
                }
            }
        }
        for (; p < K; ++p) {
            float av = Ar[p];
            const uint32_t* Bp = B+p*K_ints;
            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t pk = Bp[g];
                float* co = Cr+g*32;
                for (int b = 0; b < 32; ++b) {
                    co[b] += av*((pk&(1u<<b))?1.0f:-1.0f);
                }
            }
        }
    }
}
