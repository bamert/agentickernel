#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Calculates Matrix C = Matrix A * Matrix B
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
// ---------------------------------------------------------
// Final Optimization Strategy:
// 1. Foundation on Proven Winner: This kernel starts with the best-performing
//    structure from kernel17.
// 2. Prefetch Distance Tuning: This kernel fine-tunes the key parameter from
//    the software prefetching strategy. The prefetch distance is changed from
//    16 to 8. A shorter distance is tested to see if it better aligns with
//    the hardware's L1/L2 cache latencies and behavior.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    size_t M_main = M - (M % 3);
    const int prefetch_dist = 8; // Tuned from 16 to 8

    const uint32x4_t pos_mask0={1U<<0,1U<<1,1U<<2,1U<<3}, pos_mask1={1U<<4,1U<<5,1U<<6,1U<<7},
                     pos_mask2={1U<<8,1U<<9,1U<<10,1U<<11}, pos_mask3={1U<<12,1U<<13,1U<<14,1U<<15},
                     pos_mask4={1U<<16,1U<<17,1U<<18,1U<<19}, pos_mask5={1U<<20,1U<<21,1U<<22,1U<<23},
                     pos_mask6={1U<<24,1U<<25,1U<<26,1U<<27}, pos_mask7={1U<<28,1U<<29,1U<<30,1U<<31};
    const float32x4_t v_one = vdupq_n_f32(1.0f), v_neg_one = vdupq_n_f32(-1.0f);
    const uint32x4_t v_zero = vdupq_n_u32(0);

    for (size_t i = 0; i < M_main; i += 3) {
        const float* A_r0=A+i*K, *A_r1=A+(i+1)*K, *A_r2=A+(i+2)*K;
        float* C_r0=C+i*K, *C_r1=C+(i+1)*K, *C_r2=C+(i+2)*K;
        for (size_t j_chunk = 0; j_chunk < K_ints; ++j_chunk) {
            float32x4_t s0[8]={}, s1[8]={}, s2[8]={};
            size_t p = 0;
            if (K > prefetch_dist) {
                for (; p < K - prefetch_dist; ++p) {
                    __builtin_prefetch(&A_r0[p + prefetch_dist]);
                    __builtin_prefetch(&A_r1[p + prefetch_dist]);
                    __builtin_prefetch(&A_r2[p + prefetch_dist]);
                    __builtin_prefetch(&B[(p + prefetch_dist) * K_ints + j_chunk]);
                    const float32x4_t v_a0=vdupq_n_f32(A_r0[p]), v_a1=vdupq_n_f32(A_r1[p]), v_a2=vdupq_n_f32(A_r2[p]);
                    const uint32x4_t v_packed=vdupq_n_u32(B[p*K_ints+j_chunk]);
#define FMA3(idx, mask) do {uint32x4_t i=vandq_u32(v_packed,mask); uint32x4_t m=vcgtq_u32(i,v_zero); float32x4_t v_s=vbslq_f32(m,v_one,v_neg_one); s0[idx]=vfmaq_f32(s0[idx],v_a0,v_s); s1[idx]=vfmaq_f32(s1[idx],v_a1,v_s); s2[idx]=vfmaq_f32(s2[idx],v_a2,v_s);} while(0)
                    FMA3(0,pos_mask0);FMA3(1,pos_mask1);FMA3(2,pos_mask2);FMA3(3,pos_mask3);
                    FMA3(4,pos_mask4);FMA3(5,pos_mask5);FMA3(6,pos_mask6);FMA3(7,pos_mask7);
                }
            }
            for (; p < K; ++p) {
                const float32x4_t v_a0=vdupq_n_f32(A_r0[p]), v_a1=vdupq_n_f32(A_r1[p]), v_a2=vdupq_n_f32(A_r2[p]);
                const uint32x4_t v_packed=vdupq_n_u32(B[p*K_ints+j_chunk]);
                FMA3(0,pos_mask0);FMA3(1,pos_mask1);FMA3(2,pos_mask2);FMA3(3,pos_mask3);
                FMA3(4,pos_mask4);FMA3(5,pos_mask5);FMA3(6,pos_mask6);FMA3(7,pos_mask7);
            }
            float *c0=C_r0+j_chunk*32, *c1=C_r1+j_chunk*32, *c2=C_r2+j_chunk*32;
            vst1q_f32(c0,s0[0]);vst1q_f32(c0+4,s0[1]);vst1q_f32(c0+8,s0[2]);vst1q_f32(c0+12,s0[3]);vst1q_f32(c0+16,s0[4]);vst1q_f32(c0+20,s0[5]);vst1q_f32(c0+24,s0[6]);vst1q_f32(c0+28,s0[7]);
            vst1q_f32(c1,s1[0]);vst1q_f32(c1+4,s1[1]);vst1q_f32(c1+8,s1[2]);vst1q_f32(c1+12,s1[3]);vst1q_f32(c1+16,s1[4]);vst1q_f32(c1+20,s1[5]);vst1q_f32(c1+24,s1[6]);vst1q_f32(c1+28,s1[7]);
            vst1q_f32(c2,s2[0]);vst1q_f32(c2+4,s2[1]);vst1q_f32(c2+8,s2[2]);vst1q_f32(c2+12,s2[3]);vst1q_f32(c2+16,s2[4]);vst1q_f32(c2+20,s2[5]);vst1q_f32(c2+24,s2[6]);vst1q_f32(c2+28,s2[7]);
        }
    }
    size_t M_rem = M - M_main;
    if (M_rem >= 2) {
        size_t i = M_main;
        const float* A_r0=A+i*K, *A_r1=A+(i+1)*K; float* C_r0=C+i*K, *C_r1=C+(i+1)*K;
        for (size_t j_chunk = 0; j_chunk < K_ints; ++j_chunk) {
            float32x4_t s0[8]={}, s1[8]={};
            size_t p=0;
            if (K > prefetch_dist) {
                for(; p < K-prefetch_dist; ++p) {
                    __builtin_prefetch(&A_r0[p+prefetch_dist]); __builtin_prefetch(&A_r1[p+prefetch_dist]); __builtin_prefetch(&B[(p+prefetch_dist)*K_ints+j_chunk]);
                    const float32x4_t v_a0=vdupq_n_f32(A_r0[p]), v_a1=vdupq_n_f32(A_r1[p]);
                    const uint32x4_t v_packed=vdupq_n_u32(B[p*K_ints+j_chunk]);
#define FMA2(idx, mask) do {uint32x4_t i=vandq_u32(v_packed,mask); uint32x4_t m=vcgtq_u32(i,v_zero); float32x4_t v_s=vbslq_f32(m,v_one,v_neg_one); s0[idx]=vfmaq_f32(s0[idx],v_a0,v_s); s1[idx]=vfmaq_f32(s1[idx],v_a1,v_s);} while(0)
                    FMA2(0,pos_mask0);FMA2(1,pos_mask1);FMA2(2,pos_mask2);FMA2(3,pos_mask3);FMA2(4,pos_mask4);FMA2(5,pos_mask5);FMA2(6,pos_mask6);FMA2(7,pos_mask7);
                }
            }
            for(; p < K; ++p) {
                const float32x4_t v_a0=vdupq_n_f32(A_r0[p]), v_a1=vdupq_n_f32(A_r1[p]);
                const uint32x4_t v_packed=vdupq_n_u32(B[p*K_ints+j_chunk]);
                FMA2(0,pos_mask0);FMA2(1,pos_mask1);FMA2(2,pos_mask2);FMA2(3,pos_mask3);FMA2(4,pos_mask4);FMA2(5,pos_mask5);FMA2(6,pos_mask6);FMA2(7,pos_mask7);
            }
            float *c0=C_r0+j_chunk*32, *c1=C_r1+j_chunk*32;
            vst1q_f32(c0,s0[0]);vst1q_f32(c0+4,s0[1]);vst1q_f32(c0+8,s0[2]);vst1q_f32(c0+12,s0[3]);vst1q_f32(c0+16,s0[4]);vst1q_f32(c0+20,s0[5]);vst1q_f32(c0+24,s0[6]);vst1q_f32(c0+28,s0[7]);
            vst1q_f32(c1,s1[0]);vst1q_f32(c1+4,s1[1]);vst1q_f32(c1+8,s1[2]);vst1q_f32(c1+12,s1[3]);vst1q_f32(c1+16,s1[4]);vst1q_f32(c1+20,s1[5]);vst1q_f32(c1+24,s1[6]);vst1q_f32(c1+28,s1[7]);
        }
        M_rem -= 2; M_main +=2;
    }
    if (M_rem >= 1) {
        size_t i = M_main;
        const float* A_r0=A+i*K; float* C_r0=C+i*K;
        for (size_t j_chunk = 0; j_chunk < K_ints; ++j_chunk) {
            float32x4_t s0[8]={};
            size_t p=0;
            if (K > prefetch_dist) {
                for(; p < K-prefetch_dist; ++p) {
                    __builtin_prefetch(&A_r0[p+prefetch_dist]); __builtin_prefetch(&B[(p+prefetch_dist)*K_ints+j_chunk]);
                    const float32x4_t v_a0=vdupq_n_f32(A_r0[p]);
                    const uint32x4_t v_packed=vdupq_n_u32(B[p*K_ints+j_chunk]);
#define FMA1(idx, mask) do {uint32x4_t i=vandq_u32(v_packed,mask); uint32x4_t m=vcgtq_u32(i,v_zero); float32x4_t v_s=vbslq_f32(m,v_one,v_neg_one); s0[idx]=vfmaq_f32(s0[idx],v_a0,v_s);} while(0)
                    FMA1(0,pos_mask0);FMA1(1,pos_mask1);FMA1(2,pos_mask2);FMA1(3,pos_mask3);FMA1(4,pos_mask4);FMA1(5,pos_mask5);FMA1(6,pos_mask6);FMA1(7,pos_mask7);
                }
            }
            for(; p < K; ++p) {
                const float32x4_t v_a0=vdupq_n_f32(A_r0[p]);
                const uint32x4_t v_packed=vdupq_n_u32(B[p*K_ints+j_chunk]);
                FMA1(0,pos_mask0);FMA1(1,pos_mask1);FMA1(2,pos_mask2);FMA1(3,pos_mask3);FMA1(4,pos_mask4);FMA1(5,pos_mask5);FMA1(6,pos_mask6);FMA1(7,pos_mask7);
            }
            float *c0=C_r0+j_chunk*32;
            vst1q_f32(c0,s0[0]);vst1q_f32(c0+4,s0[1]);vst1q_f32(c0+8,s0[2]);vst1q_f32(c0+12,s0[3]);
            vst1q_f32(c0+16,s0[4]);vst1q_f32(c0+20,s0[5]);vst1q_f32(c0+24,s0[6]);vst1q_f32(c0+28,s0[7]);
        }
    }
}
