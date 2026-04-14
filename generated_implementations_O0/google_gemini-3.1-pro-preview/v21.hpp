#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    std::vector<float> row_sums(M, 0.0f);
    for (size_t i = 0; i < M; ++i) {
        float sum = 0;
        for (size_t p = 0; p < K; ++p) sum += A[i * K + p];
        row_sums[i] = sum;
    }

    constexpr size_t P_BLK = 256; 
    
    for (size_t j_int = 0; j_int < K_ints; ++j_int) {
        for (size_t p_base = 0; p_base < K; p_base += P_BLK) {
            size_t p_end = (p_base + P_BLK < K) ? p_base + P_BLK : K;
            
            // Decode B block (transpose the bits into float matrix 32 x P_BLK)
            // b_f[b][p_in] is more SIMD friendly for the inner loop over p!
            // Wait, if inner loop is b, b_f[p_in][b] is better.
            // But if we want to vectorize over 'b', b_f[p_in][b] should be contiguous.
            alignas(64) float b_f[P_BLK][32];
            for (size_t p = p_base; p < p_end; ++p) {
                uint32_t pk = B[p * K_ints + j_int];
                size_t p_in = p - p_base;
                #pragma GCC unroll 32
                for (int b = 0; b < 32; ++b) {
                    b_f[p_in][b] = (pk & (1u << b)) ? 1.0f : 0.0f;
                }
            }

            size_t i = 0;
            // Process M in blocks of 4
            for (; i + 3 < M; i += 4) {
                float c0[32] = {0}; float c1[32] = {0};
                float c2[32] = {0}; float c3[32] = {0};
                
                for (size_t p = p_base; p < p_end; ++p) {
                    size_t p_in = p - p_base;
                    float a0 = A[(i + 0) * K + p];
                    float a1 = A[(i + 1) * K + p];
                    float a2 = A[(i + 2) * K + p];
                    float a3 = A[(i + 3) * K + p];
                    
                    // The compiler vectorizes this inner loop over `b` perfectly.
                    #pragma GCC unroll 8
                    for (int b = 0; b < 32; ++b) {
                        float fb = b_f[p_in][b];
                        c0[b] += a0 * fb;
                        c1[b] += a1 * fb;
                        c2[b] += a2 * fb;
                        c3[b] += a3 * fb;
                    }
                }
                
                for (int b = 0; b < 32; ++b) {
                    if (p_base == 0) {
                        C[(i + 0) * K + j_int * 32 + b] = c0[b];
                        C[(i + 1) * K + j_int * 32 + b] = c1[b];
                        C[(i + 2) * K + j_int * 32 + b] = c2[b];
                        C[(i + 3) * K + j_int * 32 + b] = c3[b];
                    } else {
                        C[(i + 0) * K + j_int * 32 + b] += c0[b];
                        C[(i + 1) * K + j_int * 32 + b] += c1[b];
                        C[(i + 2) * K + j_int * 32 + b] += c2[b];
                        C[(i + 3) * K + j_int * 32 + b] += c3[b];
                    }
                }
            }
            for (; i < M; ++i) {
                float c0[32] = {0};
                for (size_t p = p_base; p < p_end; ++p) {
                    size_t p_in = p - p_base;
                    float a0 = A[i * K + p];
                    for (int b = 0; b < 32; ++b) {
                        c0[b] += a0 * b_f[p_in][b];
                    }
                }
                for (int b = 0; b < 32; ++b) {
                    if (p_base == 0) {
                        C[i * K + j_int * 32 + b] = c0[b];
                    } else {
                        C[i * K + j_int * 32 + b] += c0[b];
                    }
                }
            }
        }
        
        // Finalize C columns for this j_int
        for (size_t i = 0; i < M; ++i) {
            for (int b = 0; b < 32; ++b) {
                C[i * K + j_int * 32 + b] = 2.0f * C[i * K + j_int * 32 + b] - row_sums[i];
            }
        }
    }
}
