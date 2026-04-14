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
// Optimization Strategy:
// 1. Balanced Tiling and Unrolling: This kernel finds the optimal balance to
//    prevent register spilling, which hindered kernel14. It uses MR=2 tiling instead
//    of MR=3, freeing up enough registers to safely apply K-dimension unrolling.
// 2. Optimal Register Pressure: The MR=2 tiling combined with K-unrolling of 2
//    fits comfortably within the 32 NEON vector registers, avoiding slow
//    stack spills in the inner loop.
// 3. Maximum Instruction-Level Parallelism: By combining the amortization from
//    tiling with the reduced loop overhead from unrolling, this kernel maximizes
//    the amount of useful FMA work the CPU can perform per cycle.
// 4. Fully Specialized Code Paths: Both the main MR=2 loop and the MR=1 cleanup
//    loop use K-dimension unrolling, ensuring peak performance for all rows.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    size_t M_main_2 = M - (M % 2);
    size_t K_main_2 = K - (K % 2);

    const uint32x4_t pos_mask0={1U<<0,1U<<1,1U<<2,1U<<3}, pos_mask1={1U<<4,1U<<5,1U<<6,1U<<7},
                     pos_mask2={1U<<8,1U<<9,1U<<10,1U<<11}, pos_mask3={1U<<12,1U<<13,1U<<14,1U<<15},
                     pos_mask4={1U<<16,1U<<17,1U<<18,1U<<19}, pos_mask5={1U<<20,1U<<21,1U<<22,1U<<23},
                     pos_mask6={1U<<24,1U<<25,1U<<26,1U<<27}, pos_mask7={1U<<28,1U<<29,1U<<30,1U<<31};
    const float32x4_t v_one = vdupq_n_f32(1.0f), v_neg_one = vdupq_n_f32(-1.0f);
    const uint32x4_t v_zero = vdupq_n_u32(0);

    for (size_t i = 0; i < M_main_2; i += 2) {
        const float* A_r0=A+i*K, *A_r1=A+(i+1)*K; float* C_r0=C+i*K, *C_r1=C+(i+1)*K;
        for (size_t j_chunk = 0; j_chunk < K_ints; ++j_chunk) {
            float32x4_t s0[8]={}, s1[8]={};
#define FMA_BLOCK_2(V_A0, V_A1, V_PACKED) do { \
    uint32x4_t m, iso; float32x4_t v_s; \
    iso=vandq_u32(V_PACKED,pos_mask0);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[0]=vfmaq_f32(s0[0],V_A0,v_s);s1[0]=vfmaq_f32(s1[0],V_A1,v_s); \
    iso=vandq_u32(V_PACKED,pos_mask1);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[1]=vfmaq_f32(s0[1],V_A0,v_s);s1[1]=vfmaq_f32(s1[1],V_A1,v_s); \
    iso=vandq_u32(V_PACKED,pos_mask2);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[2]=vfmaq_f32(s0[2],V_A0,v_s);s1[2]=vfmaq_f32(s1[2],V_A1,v_s); \
    iso=vandq_u32(V_PACKED,pos_mask3);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[3]=vfmaq_f32(s0[3],V_A0,v_s);s1[3]=vfmaq_f32(s1[3],V_A1,v_s); \
    iso=vandq_u32(V_PACKED,pos_mask4);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[4]=vfmaq_f32(s0[4],V_A0,v_s);s1[4]=vfmaq_f32(s1[4],V_A1,v_s); \
    iso=vandq_u32(V_PACKED,pos_mask5);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[5]=vfmaq_f32(s0[5],V_A0,v_s);s1[5]=vfmaq_f32(s1[5],V_A1,v_s); \
    iso=vandq_u32(V_PACKED,pos_mask6);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[6]=vfmaq_f32(s0[6],V_A0,v_s);s1[6]=vfmaq_f32(s1[6],V_A1,v_s); \
    iso=vandq_u32(V_PACKED,pos_mask7);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[7]=vfmaq_f32(s0[7],V_A0,v_s);s1[7]=vfmaq_f32(s1[7],V_A1,v_s); \
} while(0)
            for (size_t p=0; p < K_main_2; p+=2) {
                const float32x4_t v_a0_p0=vdupq_n_f32(A_r0[p]), v_a1_p0=vdupq_n_f32(A_r1[p]);
                const uint32x4_t v_packed_p0=vdupq_n_u32(B[p*K_ints+j_chunk]);
                FMA_BLOCK_2(v_a0_p0, v_a1_p0, v_packed_p0);
                const float32x4_t v_a0_p1=vdupq_n_f32(A_r0[p+1]), v_a1_p1=vdupq_n_f32(A_r1[p+1]);
                const uint32x4_t v_packed_p1=vdupq_n_u32(B[(p+1)*K_ints+j_chunk]);
                FMA_BLOCK_2(v_a0_p1, v_a1_p1, v_packed_p1);
            }
            if (K % 2) {
                size_t p = K_main_2;
                const float32x4_t v_a0=vdupq_n_f32(A_r0[p]), v_a1=vdupq_n_f32(A_r1[p]);
                const uint32x4_t v_packed=vdupq_n_u32(B[p*K_ints+j_chunk]);
                FMA_BLOCK_2(v_a0, v_a1, v_packed);
            }
            float *c0=C_r0+j_chunk*32, *c1=C_r1+j_chunk*32;
            vst1q_f32(c0,s0[0]);vst1q_f32(c0+4,s0[1]);vst1q_f32(c0+8,s0[2]);vst1q_f32(c0+12,s0[3]);vst1q_f32(c0+16,s0[4]);vst1q_f32(c0+20,s0[5]);vst1q_f32(c0+24,s0[6]);vst1q_f32(c0+28,s0[7]);
            vst1q_f32(c1,s1[0]);vst1q_f32(c1+4,s1[1]);vst1q_f32(c1+8,s1[2]);vst1q_f32(c1+12,s1[3]);vst1q_f32(c1+16,s1[4]);vst1q_f32(c1+20,s1[5]);vst1q_f32(c1+24,s1[6]);vst1q_f32(c1+28,s1[7]);
        }
    }
    if (M % 2) {
        size_t i = M_main_2;
        const float* A_r0=A+i*K; float* C_r0=C+i*K;
        for (size_t j_chunk = 0; j_chunk < K_ints; ++j_chunk) {
            float32x4_t s0[8]={};
#define FMA_BLOCK_1(V_A0, V_PACKED) do { \
    uint32x4_t m, iso; float32x4_t v_s; \
    iso=vandq_u32(V_PACKED,pos_mask0);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[0]=vfmaq_f32(s0[0],V_A0,v_s); \
    iso=vandq_u32(V_PACKED,pos_mask1);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[1]=vfmaq_f32(s0[1],V_A0,v_s); \
    iso=vandq_u32(V_PACKED,pos_mask2);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[2]=vfmaq_f32(s0[2],V_A0,v_s); \
    iso=vandq_u32(V_PACKED,pos_mask3);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[3]=vfmaq_f32(s0[3],V_A0,v_s); \
    iso=vandq_u32(V_PACKED,pos_mask4);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[4]=vfmaq_f32(s0[4],V_A0,v_s); \
    iso=vandq_u32(V_PACKED,pos_mask5);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[5]=vfmaq_f32(s0[5],V_A0,v_s); \
    iso=vandq_u32(V_PACKED,pos_mask6);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[6]=vfmaq_f32(s0[6],V_A0,v_s); \
    iso=vandq_u32(V_PACKED,pos_mask7);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[7]=vfmaq_f32(s0[7],V_A0,v_s); \
} while(0)
            for (size_t p=0; p < K_main_2; p+=2) {
                const float32x4_t v_a0_p0=vdupq_n_f32(A_r0[p]);
                const uint32x4_t v_packed_p0=vdupq_n_u32(B[p*K_ints+j_chunk]);
                FMA_BLOCK_1(v_a0_p0, v_packed_p0);
                const float32x4_t v_a0_p1=vdupq_n_f32(A_r0[p+1]);
                const uint32x4_t v_packed_p1=vdupq_n_u32(B[(p+1)*K_ints+j_chunk]);
                FMA_BLOCK_1(v_a0_p1, v_packed_p1);
            }
            if (K % 2) {
                size_t p = K_main_2;
                const float32x4_t v_a0=vdupq_n_f32(A_r0[p]);
                const uint32x4_t v_packed=vdupq_n_u32(B[p*K_ints+j_chunk]);
                FMA_BLOCK_1(v_a0, v_packed);
            }
            float *c0=C_r0+j_chunk*32;
            vst1q_f32(c0,s0[0]);vst1q_f32(c0+4,s0[1]);vst1q_f32(c0+8,s0[2]);vst1q_f32(c0+12,s0[3]);
            vst1q_f32(c0+16,s0[4]);vst1q_f32(c0+20,s0[5]);vst1q_f32(c0+24,s0[6]);vst1q_f32(c0+28,s0[7]);
        }
    }
}
