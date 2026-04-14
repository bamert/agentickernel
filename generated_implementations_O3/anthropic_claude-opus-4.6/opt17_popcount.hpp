#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Transpose B: B[p][j] -> B_T[j][p]
    // B_T[j * K_ints + p/32] bit (p%32) = B[p * K_ints + j/32] bit (j%32)
    uint32_t* __restrict__ B_T = (uint32_t*)__builtin_alloca(K * K_ints * sizeof(uint32_t));
    
    for (size_t idx = 0; idx < K * K_ints; ++idx) {
        B_T[idx] = 0;
    }
    
    // Transpose using 32x32 block transpose for efficiency
    for (size_t pb = 0; pb < K_ints; ++pb) {
        for (size_t jb = 0; jb < K_ints; ++jb) {
            // Transpose a 32x32 bit block: B rows [pb*32..pb*32+31], cols [jb*32..jb*32+31]
            // into B_T rows [jb*32..jb*32+31], cols [pb*32..pb*32+31]
            uint32_t block[32];
            for (int r = 0; r < 32; ++r) {
                block[r] = B[(pb * 32 + r) * K_ints + jb];
            }
            // Bit-transpose this 32x32 block
            // Result: out[c] has bit r set if block[r] has bit c set
            for (int bit = 0; bit < 32; ++bit) {
                uint32_t col = 0;
                for (int r = 0; r < 32; ++r) {
                    col |= ((block[r] >> bit) & 1) << r;
                }
                B_T[(jb * 32 + bit) * K_ints + pb] = col;
            }
        }
    }

    // Now for each (i, j): C[i][j] = sum_p A[i][p] * sign(B_T[j][p])
    // = 2 * sum_{p: B_T[j][p]=1} A[i][p] - rowsum_i
    //
    // Precompute partial sums of A in groups of 32
    // partial_sum[i][g] = sum of A[i][g*32 .. g*32+31]
    float* __restrict__ partial_sums = (float*)__builtin_alloca(M * K_ints * sizeof(float));
    float* __restrict__ row_sums = (float*)__builtin_alloca(M * sizeof(float));
    
    for (size_t i = 0; i < M; ++i) {
        float rsum = 0.0f;
        for (size_t g = 0; g < K_ints; ++g) {
            float32x4_t s = vdupq_n_f32(0.0f);
            const float* ap = A + i * K + g * 32;
            for (int k = 0; k < 32; k += 4) {
                s = vaddq_f32(s, vld1q_f32(ap + k));
            }
            float gsum = vaddvq_f32(s);
            partial_sums[i * K_ints + g] = gsum;
            rsum += gsum;
        }
        row_sums[i] = rsum;
    }

    // For each j, compute C[i][j] for all i
    // C[i][j] = 2 * sum_g (sum of A[i][g*32+p] where B_T[j][g] bit p is set) - rowsum[i]
    // 
    // For each group g, we need: sum of A[i][g*32+p] for set bits in B_T[j][g]
    // If ALL bits set: sum = partial_sums[i][g]
    // If NO bits set: sum = 0
    // Otherwise: popcount approach: 
    //   sum_set = (partial_sums[i][g] + dot_with_signs) / 2 where dot_with_signs uses popcount
    //   Actually: sum_set = (partial_sums[i][g] + sum_set - sum_unset) / 2... circular.
    //
    // Better: we can compute the "signed dot product" directly.
    // signed_dot = sum_g popcount_signed(A[i][g*32..], B_T[j][g])
    // where popcount_signed uses the sign-XOR approach from opt11.
    
    // Actually let's just use the sign-XOR approach with B_T.
    // For each j, B_T[j] is a packed row of K bits.
    // For each i, process all groups.
    
    // Nibble LUT
    uint32_t sign_lut[16 * 4] __attribute__((aligned(16)));
    for (int nibble = 0; nibble < 16; ++nibble) {
        for (int b = 0; b < 4; ++b) {
            int bit = (nibble >> b) & 1;
            sign_lut[nibble * 4 + b] = bit ? 0u : 0x80000000u;
        }
    }

    // Inner product approach: for each (i,j), accumulate over p
    // This gives us perfect C locality (one element at a time) but bad B_T locality
    // unless we process multiple j values together.
    
    // Process 4 i rows at a time, iterate over j, accumulate using sign-XOR
    for (size_t i = 0; i < M; i += 4) {
        size_t i_end = i + 4;
        if (i_end > M) i_end = M;
        size_t ni = i_end - i;
        
        for (size_t j = 0; j < K; ++j) {
            const uint32_t* bt_row = B_T + j * K_ints;
            
            float32x4_t acc[4];
            for (size_t r = 0; r < ni; ++r) acc[r] = vdupq_n_f32(0.0f);
            
            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t packed = bt_row[g];
                
                for (int n = 0; n < 8; ++n) {
                    int nib = (packed >> (n * 4)) & 0xF;
                    uint32x4_t mask = vld1q_u32(&sign_lut[nib * 4]);
                    
                    for (size_t r = 0; r < ni; ++r) {
                        const float* ap = A + (i + r) * K + g * 32 + n * 4;
                        uint32x4_t a_bits = vreinterpretq_u32_f32(vld1q_f32(ap));
                        acc[r] = vaddq_f32(acc[r], vreinterpretq_f32_u32(veorq_u32(a_bits, mask)));
                    }
                }
            }
            
            for (size_t r = 0; r < ni; ++r) {
                C[(i + r) * K + j] = vaddvq_f32(acc[r]);
            }
        }
    }
}
