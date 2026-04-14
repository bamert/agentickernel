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
// 1. Corrected Implementation: This kernel fixes the compilation errors from
//    kernel13 by properly scoping variables within the unrolled loop blocks.
// 2. Combined Tiling & Unrolling: It retains the most advanced strategy:
//    - Tiling in the M-dimension (MR=3, 2, 1) to amortize sign-generation cost.
//    - Unrolling the innermost K-dimension loop (p-loop) by 2 to reduce loop
//      overhead and increase instruction-level parallelism.
// 3. Fully Optimized Cleanup: The cleanup logic for both M and K dimensions
//    uses specialized, high-performance code, ensuring no part of the matrix
//    is processed by a slow, generic loop.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    size_t M_main_3 = M - (M % 3);
    size_t K_main_2 = K - (K % 2);

    const uint32x4_t pos_mask0 = {1U<<0,1U<<1,1U<<2,1U<<3}, pos_mask1 = {1U<<4,1U<<5,1U<<6,1U<<7},
                     pos_mask2 = {1U<<8,1U<<9,1U<<10,1U<<11}, pos_mask3 = {1U<<12,1U<<13,1U<<14,1U<<15},
                     pos_mask4 = {1U<<16,1U<<17,1U<<18,1U<<19}, pos_mask5 = {1U<<20,1U<<21,1U<<22,1U<<23},
                     pos_mask6 = {1U<<24,1U<<25,1U<<26,1U<<27}, pos_mask7 = {1U<<28,1U<<29,1U<<30,1U<<31};
    const float32x4_t v_one = vdupq_n_f32(1.0f), v_neg_one = vdupq_n_f32(-1.0f);
    const uint32x4_t v_zero = vdupq_n_u32(0);

    // Main loop: MR=3 and K unrolled by 2
    for (size_t i = 0; i < M_main_3; i += 3) {
        const float* A_r0=A+i*K, *A_r1=A+(i+1)*K, *A_r2=A+(i+2)*K;
        float* C_r0=C+i*K, *C_r1=C+(i+1)*K, *C_r2=C+(i+2)*K;
        for (size_t j_chunk = 0; j_chunk < K_ints; ++j_chunk) {
            float32x4_t s0[8]={}, s1[8]={}, s2[8]={};
            
#define FMA_BLOCK_3(V_A0, V_A1, V_A2, V_PACKED) do { \
    uint32x4_t m, iso; float32x4_t v_s; \
    iso=vandq_u32(V_PACKED,pos_mask0);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[0]=vfmaq_f32(s0[0],V_A0,v_s);s1[0]=vfmaq_f32(s1[0],V_A1,v_s);s2[0]=vfmaq_f32(s2[0],V_A2,v_s); \
    iso=vandq_u32(V_PACKED,pos_mask1);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[1]=vfmaq_f32(s0[1],V_A0,v_s);s1[1]=vfmaq_f32(s1[1],V_A1,v_s);s2[1]=vfmaq_f32(s2[1],V_A2,v_s); \
    iso=vandq_u32(V_PACKED,pos_mask2);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[2]=vfmaq_f32(s0[2],V_A0,v_s);s1[2]=vfmaq_f32(s1[2],V_A1,v_s);s2[2]=vfmaq_f32(s2[2],V_A2,v_s); \
    iso=vandq_u32(V_PACKED,pos_mask3);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[3]=vfmaq_f32(s0[3],V_A0,v_s);s1[3]=vfmaq_f32(s1[3],V_A1,v_s);s2[3]=vfmaq_f32(s2[3],V_A2,v_s); \
    iso=vandq_u32(V_PACKED,pos_mask4);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[4]=vfmaq_f32(s0[4],V_A0,v_s);s1[4]=vfmaq_f32(s1[4],V_A1,v_s);s2[4]=vfmaq_f32(s2[4],V_A2,v_s); \
    iso=vandq_u32(V_PACKED,pos_mask5);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[5]=vfmaq_f32(s0[5],V_A0,v_s);s1[5]=vfmaq_f32(s1[5],V_A1,v_s);s2[5]=vfmaq_f32(s2[5],V_A2,v_s); \
    iso=vandq_u32(V_PACKED,pos_mask6);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[6]=vfmaq_f32(s0[6],V_A0,v_s);s1[6]=vfmaq_f32(s1[6],V_A1,v_s);s2[6]=vfmaq_f32(s2[6],V_A2,v_s); \
    iso=vandq_u32(V_PACKED,pos_mask7);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[7]=vfmaq_f32(s0[7],V_A0,v_s);s1[7]=vfmaq_f32(s1[7],V_A1,v_s);s2[7]=vfmaq_f32(s2[7],V_A2,v_s); \
} while(0)
            
            for (size_t p = 0; p < K_main_2; p += 2) {
                const float32x4_t v_a0_p0=vdupq_n_f32(A_r0[p]), v_a1_p0=vdupq_n_f32(A_r1[p]), v_a2_p0=vdupq_n_f32(A_r2[p]);
                const uint32x4_t v_packed_p0=vdupq_n_u32(B[p*K_ints+j_chunk]);
                FMA_BLOCK_3(v_a0_p0, v_a1_p0, v_a2_p0, v_packed_p0);
                
                const float32x4_t v_a0_p1=vdupq_n_f32(A_r0[p+1]), v_a1_p1=vdupq_n_f32(A_r1[p+1]), v_a2_p1=vdupq_n_f32(A_r2[p+1]);
                const uint32x4_t v_packed_p1=vdupq_n_u32(B[(p+1)*K_ints+j_chunk]);
                FMA_BLOCK_3(v_a0_p1, v_a1_p1, v_a2_p1, v_packed_p1);
            }
            if (K % 2) {
                size_t p = K_main_2;
                const float32x4_t v_a0=vdupq_n_f32(A_r0[p]), v_a1=vdupq_n_f32(A_r1[p]), v_a2=vdupq_n_f32(A_r2[p]);
                const uint32x4_t v_packed=vdupq_n_u32(B[p*K_ints+j_chunk]);
                FMA_BLOCK_3(v_a0, v_a1, v_a2, v_packed);
            }
            float *c0=C_r0+j_chunk*32, *c1=C_r1+j_chunk*32, *c2=C_r2+j_chunk*32;
            vst1q_f32(c0,s0[0]);vst1q_f32(c0+4,s0[1]);vst1q_f32(c0+8,s0[2]);vst1q_f32(c0+12,s0[3]);vst1q_f32(c0+16,s0[4]);vst1q_f32(c0+20,s0[5]);vst1q_f32(c0+24,s0[6]);vst1q_f32(c0+28,s0[7]);
            vst1q_f32(c1,s1[0]);vst1q_f32(c1+4,s1[1]);vst1q_f32(c1+8,s1[2]);vst1q_f32(c1+12,s1[3]);vst1q_f32(c1+16,s1[4]);vst1q_f32(c1+20,s1[5]);vst1q_f32(c1+24,s1[6]);vst1q_f32(c1+28,s1[7]);
            vst1q_f32(c2,s2[0]);vst1q_f32(c2+4,s2[1]);vst1q_f32(c2+8,s2[2]);vst1q_f32(c2+12,s2[3]);vst1q_f32(c2+16,s2[4]);vst1q_f32(c2+20,s2[5]);vst1q_f32(c2+24,s2[6]);vst1q_f32(c2+28,s2[7]);
        }
    }
    size_t M_rem_start = M_main_3;
    if (M - M_rem_start >= 2) {
        size_t i = M_rem_start;
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
        M_rem_start += 2;
    }
    if (M - M_rem_start >= 1) {
        size_t i = M_rem_start;
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
