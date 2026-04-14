
#pragma once
#include <cstdint>
#include <cstddef>
#include <string.h>

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, 
            float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    size_t i = 0;
    for (; i + 4 <= M; i += 4) {
        float* __restrict__ C0 = C + (i+0)*K;
        float* __restrict__ C1 = C + (i+1)*K;
        float* __restrict__ C2 = C + (i+2)*K;
        float* __restrict__ C3 = C + (i+3)*K;
        const float* __restrict__ A0 = A + (i+0)*K;
        const float* __restrict__ A1 = A + (i+1)*K;
        const float* __restrict__ A2 = A + (i+2)*K;
        const float* __restrict__ A3 = A + (i+3)*K;

        memset(C0, 0, K*sizeof(float));
        memset(C1, 0, K*sizeof(float));
        memset(C2, 0, K*sizeof(float));
        memset(C3, 0, K*sizeof(float));

        size_t p = 0;
        for (; p + 12 <= K; p += 12) {
            float a0_0=A0[p+0],a0_1=A0[p+1],a0_2=A0[p+2],a0_3=A0[p+3],a0_4=A0[p+4],a0_5=A0[p+5];
            float a0_6=A0[p+6],a0_7=A0[p+7],a0_8=A0[p+8],a0_9=A0[p+9],a0_a=A0[p+10],a0_b=A0[p+11];
            float a1_0=A1[p+0],a1_1=A1[p+1],a1_2=A1[p+2],a1_3=A1[p+3],a1_4=A1[p+4],a1_5=A1[p+5];
            float a1_6=A1[p+6],a1_7=A1[p+7],a1_8=A1[p+8],a1_9=A1[p+9],a1_a=A1[p+10],a1_b=A1[p+11];
            float a2_0=A2[p+0],a2_1=A2[p+1],a2_2=A2[p+2],a2_3=A2[p+3],a2_4=A2[p+4],a2_5=A2[p+5];
            float a2_6=A2[p+6],a2_7=A2[p+7],a2_8=A2[p+8],a2_9=A2[p+9],a2_a=A2[p+10],a2_b=A2[p+11];
            float a3_0=A3[p+0],a3_1=A3[p+1],a3_2=A3[p+2],a3_3=A3[p+3],a3_4=A3[p+4],a3_5=A3[p+5];
            float a3_6=A3[p+6],a3_7=A3[p+7],a3_8=A3[p+8],a3_9=A3[p+9],a3_a=A3[p+10],a3_b=A3[p+11];

            const uint32_t *Bp0=B+(p+0)*K_ints,*Bp1=B+(p+1)*K_ints,*Bp2=B+(p+2)*K_ints,*Bp3=B+(p+3)*K_ints;
            const uint32_t *Bp4=B+(p+4)*K_ints,*Bp5=B+(p+5)*K_ints,*Bp6=B+(p+6)*K_ints,*Bp7=B+(p+7)*K_ints;
            const uint32_t *Bp8=B+(p+8)*K_ints,*Bp9=B+(p+9)*K_ints,*Bpa=B+(p+10)*K_ints,*Bpb=B+(p+11)*K_ints;

            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t pk0=Bp0[g],pk1=Bp1[g],pk2=Bp2[g],pk3=Bp3[g];
                uint32_t pk4=Bp4[g],pk5=Bp5[g],pk6=Bp6[g],pk7=Bp7[g];
                uint32_t pk8=Bp8[g],pk9=Bp9[g],pka=Bpa[g],pkb=Bpb[g];

                float* c0=C0+g*32,*c1=C1+g*32,*c2=C2+g*32,*c3=C3+g*32;

                for (int b = 0; b < 32; ++b) {
                    uint32_t mask = 1u << b;
                    float s0=(pk0&mask)?1.f:-1.f, s1=(pk1&mask)?1.f:-1.f;
                    float s2=(pk2&mask)?1.f:-1.f, s3=(pk3&mask)?1.f:-1.f;
                    float s4=(pk4&mask)?1.f:-1.f, s5=(pk5&mask)?1.f:-1.f;
                    float s6=(pk6&mask)?1.f:-1.f, s7=(pk7&mask)?1.f:-1.f;
                    float s8=(pk8&mask)?1.f:-1.f, s9=(pk9&mask)?1.f:-1.f;
                    float sa=(pka&mask)?1.f:-1.f, sb=(pkb&mask)?1.f:-1.f;

                    c0[b] += a0_0*s0+a0_1*s1+a0_2*s2+a0_3*s3+a0_4*s4+a0_5*s5+a0_6*s6+a0_7*s7+a0_8*s8+a0_9*s9+a0_a*sa+a0_b*sb;
                    c1[b] += a1_0*s0+a1_1*s1+a1_2*s2+a1_3*s3+a1_4*s4+a1_5*s5+a1_6*s6+a1_7*s7+a1_8*s8+a1_9*s9+a1_a*sa+a1_b*sb;
                    c2[b] += a2_0*s0+a2_1*s1+a2_2*s2+a2_3*s3+a2_4*s4+a2_5*s5+a2_6*s6+a2_7*s7+a2_8*s8+a2_9*s9+a2_a*sa+a2_b*sb;
                    c3[b] += a3_0*s0+a3_1*s1+a3_2*s2+a3_3*s3+a3_4*s4+a3_5*s5+a3_6*s6+a3_7*s7+a3_8*s8+a3_9*s9+a3_a*sa+a3_b*sb;
                }
            }
        }
        // Remainder with p=1
        for (; p < K; ++p) {
            float a0v=A0[p],a1v=A1[p],a2v=A2[p],a3v=A3[p];
            const uint32_t* Bp=B+p*K_ints;
            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t pk=Bp[g];
                float *c0=C0+g*32,*c1=C1+g*32,*c2=C2+g*32,*c3=C3+g*32;
                for (int b = 0; b < 32; ++b) {
                    float s=(pk&(1u<<b))?1.f:-1.f;
                    c0[b]+=a0v*s; c1[b]+=a1v*s; c2[b]+=a2v*s; c3[b]+=a3v*s;
                }
            }
        }
    }

    for (; i < M; ++i) {
        float* Cr=C+i*K;
        const float* Ar=A+i*K;
        memset(Cr, 0, K*sizeof(float));
        for (size_t p = 0; p < K; ++p) {
            float av=Ar[p];
            const uint32_t* Bp=B+p*K_ints;
            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t pk=Bp[g];
                float* co=Cr+g*32;
                for (int b = 0; b < 32; ++b) co[b]+=av*((pk&(1u<<b))?1.f:-1.f);
            }
        }
    }
}
