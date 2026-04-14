
#pragma once
#include <cstdint>
#include <cstddef>
#include <string.h>
#include <arm_neon.h>

// Same as v22 (M=4, p=8 explicit) but use bit math instead of ternary for sign.
// Extract bit, shift to sign position of float, XOR with 1.0f representation.
// 1.0f = 0x3F800000. Sign bit is bit 31.
// If we extract the bit and put it in bit 31, then XOR with 0x3F800000:
//   bit=0 -> XOR 0x00000000 with 0x3F800000 = 0x3F800000 = 1.0f  WRONG (should be -1.0f)
//   bit=1 -> XOR 0x80000000 with 0x3F800000 = 0xBF800000 = -1.0f WRONG
// 
// Actually: bit=1 means +1.0f, bit=0 means -1.0f.
// So: extract bit, if bit=0 we want 0x80000000 XOR with 0x3F800000 = 0xBF800000 = -1.0f
//     if bit=1 we want 0x00000000 XOR with 0x3F800000 = 0x3F800000 = +1.0f
// So: sign_xor = (~bit & 1) << 31 = (1-bit) << 31
// Or equivalently: sign_xor = (bit ^ 1) << 31

// Simpler: just use (bit << 31) ^ 0x80000000, then reinterpret XOR with 0x3F800000
// bit=0: 0 ^ 0x80000000 = 0x80000000, then 0x80000000 ^ 0x3F800000 = 0xBF800000 = -1.0f ✓
// bit=1: 0x80000000 ^ 0x80000000 = 0, then 0 ^ 0x3F800000 = 0x3F800000 = +1.0f ✓

// Or even simpler, note that for the final computation we just need a_val * sign.
// a_val XOR sign_bit gives -a_val. So if bit=0, XOR sign_bit with a_val; if bit=1, keep a_val.
// mask = ((~packed >> b) & 1) << 31 = sign bit to flip

// Let me just keep it simple and see if the compiler handles the union-based approach better.

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
        for (; p + 8 <= K; p += 8) {
            float a00=A0[p],a01=A0[p+1],a02=A0[p+2],a03=A0[p+3],a04=A0[p+4],a05=A0[p+5],a06=A0[p+6],a07=A0[p+7];
            float a10=A1[p],a11=A1[p+1],a12=A1[p+2],a13=A1[p+3],a14=A1[p+4],a15=A1[p+5],a16=A1[p+6],a17=A1[p+7];
            float a20=A2[p],a21=A2[p+1],a22=A2[p+2],a23=A2[p+3],a24=A2[p+4],a25=A2[p+5],a26=A2[p+6],a27=A2[p+7];
            float a30=A3[p],a31=A3[p+1],a32=A3[p+2],a33=A3[p+3],a34=A3[p+4],a35=A3[p+5],a36=A3[p+6],a37=A3[p+7];
            
            const uint32_t *Bp0=B+(p+0)*K_ints,*Bp1=B+(p+1)*K_ints,*Bp2=B+(p+2)*K_ints,*Bp3=B+(p+3)*K_ints;
            const uint32_t *Bp4=B+(p+4)*K_ints,*Bp5=B+(p+5)*K_ints,*Bp6=B+(p+6)*K_ints,*Bp7=B+(p+7)*K_ints;

            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t pk0=Bp0[g],pk1=Bp1[g],pk2=Bp2[g],pk3=Bp3[g];
                uint32_t pk4=Bp4[g],pk5=Bp5[g],pk6=Bp6[g],pk7=Bp7[g];
                
                float* c0=C0+g*32,*c1=C1+g*32,*c2=C2+g*32,*c3=C3+g*32;

                // Use popcount-style: for each bit position b,
                // compute contribution using add/sub instead of multiply
                for (int b = 0; b < 32; ++b) {
                    uint32_t mask = 1u << b;
                    // Instead of multiply, use conditional add/sub
                    // sum_pos = sum of a values where bit=1
                    // sum_neg = sum of a values where bit=0
                    // result = sum_pos - sum_neg = 2*sum_pos - sum_all
                    
                    float v0 = 0, v1 = 0, v2 = 0, v3 = 0;
                    
                    if (pk0 & mask) { v0+=a00; v1+=a10; v2+=a20; v3+=a30; } 
                    else            { v0-=a00; v1-=a10; v2-=a20; v3-=a30; }
                    if (pk1 & mask) { v0+=a01; v1+=a11; v2+=a21; v3+=a31; } 
                    else            { v0-=a01; v1-=a11; v2-=a21; v3-=a31; }
                    if (pk2 & mask) { v0+=a02; v1+=a12; v2+=a22; v3+=a32; } 
                    else            { v0-=a02; v1-=a12; v2-=a22; v3-=a32; }
                    if (pk3 & mask) { v0+=a03; v1+=a13; v2+=a23; v3+=a33; } 
                    else            { v0-=a03; v1-=a13; v2-=a23; v3-=a33; }
                    if (pk4 & mask) { v0+=a04; v1+=a14; v2+=a24; v3+=a34; } 
                    else            { v0-=a04; v1-=a14; v2-=a24; v3-=a34; }
                    if (pk5 & mask) { v0+=a05; v1+=a15; v2+=a25; v3+=a35; } 
                    else            { v0-=a05; v1-=a15; v2-=a25; v3-=a35; }
                    if (pk6 & mask) { v0+=a06; v1+=a16; v2+=a26; v3+=a36; } 
                    else            { v0-=a06; v1-=a16; v2-=a26; v3-=a36; }
                    if (pk7 & mask) { v0+=a07; v1+=a17; v2+=a27; v3+=a37; } 
                    else            { v0-=a07; v1-=a17; v2-=a27; v3-=a37; }
                    
                    c0[b] += v0;
                    c1[b] += v1;
                    c2[b] += v2;
                    c3[b] += v3;
                }
            }
        }
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
