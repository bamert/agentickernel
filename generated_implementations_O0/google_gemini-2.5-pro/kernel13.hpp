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
// 1. Final Refinement: This kernel builds on the successful MR=3/2/1 tiling
//    of kernel12.
// 2. Innermost Loop Unrolling: The innermost loop over the K-dimension (`p` loop)
//    is unrolled by a factor of 2. This halves the loop control overhead
//    (branches, increments) and increases instruction-level parallelism,
//    allowing the CPU to better hide latency by overlapping loads and arithmetic.
// 3. Complete Specialization: All code paths (MR=3, MR=2, MR=1) feature this
//    unrolling, along with a cleanup step for odd K, ensuring every element
//    is processed with maximum efficiency.
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

    for (size_t i = 0; i < M_main_3; i += 3) {
        const float* A_r0=A+i*K, *A_r1=A+(i+1)*K, *A_r2=A+(i+2)*K;
        float* C_r0=C+i*K, *C_r1=C+(i+1)*K, *C_r2=C+(i+2)*K;
        for (size_t j_chunk = 0; j_chunk < K_ints; ++j_chunk) {
            float32x4_t s0[8]={}, s1[8]={}, s2[8]={};
            for (size_t p = 0; p < K_main_2; p += 2) {
#define COMPUTE_P_BLOCK_3(p_offset) \
    const float32x4_t v_a0=vdupq_n_f32(A_r0[p+p_offset]), v_a1=vdupq_n_f32(A_r1[p+p_offset]), v_a2=vdupq_n_f32(A_r2[p+p_offset]); \
    const uint32x4_t v_packed=vdupq_n_u32(B[(p+p_offset)*K_ints+j_chunk]); \
    uint32x4_t iso, m; float32x4_t v_s; \
    iso=vandq_u32(v_packed,pos_mask0);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[0]=vfmaq_f32(s0[0],v_a0,v_s);s1[0]=vfmaq_f32(s1[0],v_a1,v_s);s2[0]=vfmaq_f32(s2[0],v_a2,v_s); \
    iso=vandq_u32(v_packed,pos_mask1);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[1]=vfmaq_f32(s0[1],v_a0,v_s);s1[1]=vfmaq_f32(s1[1],v_a1,v_s);s2[1]=vfmaq_f32(s2[1],v_a2,v_s); \
    iso=vandq_u32(v_packed,pos_mask2);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[2]=vfmaq_f32(s0[2],v_a0,v_s);s1[2]=vfmaq_f32(s1[2],v_a1,v_s);s2[2]=vfmaq_f32(s2[2],v_a2,v_s); \
    iso=vandq_u32(v_packed,pos_mask3);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[3]=vfmaq_f32(s0[3],v_a0,v_s);s1[3]=vfmaq_f32(s1[3],v_a1,v_s);s2[3]=vfmaq_f32(s2[3],v_a2,v_s); \
    iso=vandq_u32(v_packed,pos_mask4);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[4]=vfmaq_f32(s0[4],v_a0,v_s);s1[4]=vfmaq_f32(s1[4],v_a1,v_s);s2[4]=vfmaq_f32(s2[4],v_a2,v_s); \
    iso=vandq_u32(v_packed,pos_mask5);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[5]=vfmaq_f32(s0[5],v_a0,v_s);s1[5]=vfmaq_f32(s1[5],v_a1,v_s);s2[5]=vfmaq_f32(s2[5],v_a2,v_s); \
    iso=vandq_u32(v_packed,pos_mask6);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[6]=vfmaq_f32(s0[6],v_a0,v_s);s1[6]=vfmaq_f32(s1[6],v_a1,v_s);s2[6]=vfmaq_f32(s2[6],v_a2,v_s); \
    iso=vandq_u32(v_packed,pos_mask7);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[7]=vfmaq_f32(s0[7],v_a0,v_s);s1[7]=vfmaq_f32(s1[7],v_a1,v_s);s2[7]=vfmaq_f32(s2[7],v_a2,v_s);
                COMPUTE_P_BLOCK_3(0); COMPUTE_P_BLOCK_3(1);
            }
            if (K % 2) { COMPUTE_P_BLOCK_3(K_main_2); }
            float *c0=C_r0+j_chunk*32,*c1=C_r1+j_chunk*32,*c2=C_r2+j_chunk*32;
            for(int j=0;j<8;j++){vst1q_f32(c0+j*4,s0[j]);vst1q_f32(c1+j*4,s1[j]);vst1q_f32(c2+j*4,s2[j]);}
        }
    }
    size_t M_rem = M - M_main_3;
    if (M_rem >= 2) {
        size_t i = M_main_3;
        const float* A_r0=A+i*K, *A_r1=A+(i+1)*K; float* C_r0=C+i*K, *C_r1=C+(i+1)*K;
        for (size_t j_chunk = 0; j_chunk < K_ints; ++j_chunk) {
            float32x4_t s0[8]={}, s1[8]={};
            for (size_t p=0; p < K_main_2; p+=2) {
#define COMPUTE_P_BLOCK_2(p_offset) \
    const float32x4_t v_a0=vdupq_n_f32(A_r0[p+p_offset]), v_a1=vdupq_n_f32(A_r1[p+p_offset]); \
    const uint32x4_t v_packed=vdupq_n_u32(B[(p+p_offset)*K_ints+j_chunk]); \
    uint32x4_t iso,m; float32x4_t v_s; \
    iso=vandq_u32(v_packed,pos_mask0);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[0]=vfmaq_f32(s0[0],v_a0,v_s);s1[0]=vfmaq_f32(s1[0],v_a1,v_s); \
    iso=vandq_u32(v_packed,pos_mask1);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[1]=vfmaq_f32(s0[1],v_a0,v_s);s1[1]=vfmaq_f32(s1[1],v_a1,v_s); \
    iso=vandq_u32(v_packed,pos_mask2);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[2]=vfmaq_f32(s0[2],v_a0,v_s);s1[2]=vfmaq_f32(s1[2],v_a1,v_s); \
    iso=vandq_u32(v_packed,pos_mask3);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[3]=vfmaq_f32(s0[3],v_a0,v_s);s1[3]=vfmaq_f32(s1[3],v_a1,v_s); \
    iso=vandq_u32(v_packed,pos_mask4);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[4]=vfmaq_f32(s0[4],v_a0,v_s);s1[4]=vfmaq_f32(s1[4],v_a1,v_s); \
    iso=vandq_u32(v_packed,pos_mask5);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[5]=vfmaq_f32(s0[5],v_a0,v_s);s1[5]=vfmaq_f32(s1[5],v_a1,v_s); \
    iso=vandq_u32(v_packed,pos_mask6);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[6]=vfmaq_f32(s0[6],v_a0,v_s);s1[6]=vfmaq_f32(s1[6],v_a1,v_s); \
    iso=vandq_u32(v_packed,pos_mask7);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[7]=vfmaq_f32(s0[7],v_a0,v_s);s1[7]=vfmaq_f32(s1[7],v_a1,v_s);
                COMPUTE_P_BLOCK_2(0); COMPUTE_P_BLOCK_2(1);
            }
            if (K % 2) { COMPUTE_P_BLOCK_2(K_main_2); }
            float *c0=C_r0+j_chunk*32, *c1=C_r1+j_chunk*32;
            for(int j=0;j<8;j++){vst1q_f32(c0+j*4,s0[j]); vst1q_f32(c1+j*4,s1[j]);}
        }
        M_rem -= 2; M_main_3 +=2;
    }
    if (M_rem >= 1) {
        size_t i = M_main_3;
        const float* A_r0=A+i*K; float* C_r0=C+i*K;
        for (size_t j_chunk = 0; j_chunk < K_ints; ++j_chunk) {
            float32x4_t s0[8]={};
            for (size_t p=0; p < K_main_2; ++p) {
#define COMPUTE_P_BLOCK_1(p_offset) \
    const float32x4_t v_a0=vdupq_n_f32(A_r0[p+p_offset]); \
    const uint32x4_t v_packed=vdupq_n_u32(B[(p+p_offset)*K_ints+j_chunk]); \
    uint32x4_t iso,m; float32x4_t v_s; \
    iso=vandq_u32(v_packed,pos_mask0);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[0]=vfmaq_f32(s0[0],v_a0,v_s); \
    iso=vandq_u32(v_packed,pos_mask1);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[1]=vfmaq_f32(s0[1],v_a0,v_s); \
    iso=vandq_u32(v_packed,pos_mask2);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[2]=vfmaq_f32(s0[2],v_a0,v_s); \
    iso=vandq_u32(v_packed,pos_mask3);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[3]=vfmaq_f32(s0[3],v_a0,v_s); \
    iso=vandq_u32(v_packed,pos_mask4);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[4]=vfmaq_f32(s0[4],v_a0,v_s); \
    iso=vandq_u32(v_packed,pos_mask5);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[5]=vfmaq_f32(s0[5],v_a0,v_s); \
    iso=vandq_u32(v_packed,pos_mask6);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[6]=vfmaq_f32(s0[6],v_a0,v_s); \
    iso=vandq_u32(v_packed,pos_mask7);m=vcgtq_u32(iso,v_zero);v_s=vbslq_f32(m,v_one,v_neg_one);s0[7]=vfmaq_f32(s0[7],v_a0,v_s);
                COMPUTE_P_BLOCK_1(0); COMPUTE_P_BLOCK_1(1);
            }
            if (K % 2) { COMPUTE_P_BLOCK_1(K_main_2); }
            float *c0=C_r0+j_chunk*32;
            for(int j=0;j<8;j++){vst1q_f32(c0+j*4,s0[j]);}
        }
    }
}
