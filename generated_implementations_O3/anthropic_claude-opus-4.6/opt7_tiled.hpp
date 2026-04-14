#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Zero out C
    for (size_t i = 0; i < M * K; ++i) {
        C[i] = 0.0f;
    }

    // Tile over p to improve cache locality for both A and B
    // Each tile of p processes a block of rows in A and B
    const size_t P_TILE = 64;  // tile size for p dimension
    const size_t J_TILE = 128; // tile size for j dimension (in terms of floats, = 4 uint32 words)

    for (size_t i = 0; i < M; ++i) {
        float* c_row = C + i * K;
        const float* a_row = A + i * K;
        
        for (size_t jt = 0; jt < K; jt += J_TILE) {
            size_t j_end = jt + J_TILE;
            if (j_end > K) j_end = K;
            size_t jb_start = jt / 32;
            size_t jb_end = (j_end + 31) / 32;
            
            for (size_t pt = 0; pt < K; pt += P_TILE) {
                size_t p_end = pt + P_TILE;
                if (p_end > K) p_end = K;
                
                for (size_t p = pt; p < p_end; ++p) {
                    float a_val = a_row[p];
                    const uint32_t* b_row = B + p * K_ints;
                    
                    for (size_t jb = jb_start; jb < jb_end; ++jb) {
                        uint32_t packed = b_row[jb];
                        size_t j_base = jb * 32;
                        float* c_ptr = c_row + j_base;
                        
                        for (int bit = 0; bit < 32; ++bit) {
                            float sign = (packed & 1) ? 1.0f : -1.0f;
                            c_ptr[bit] += a_val * sign;
                            packed >>= 1;
                        }
                    }
                }
            }
        }
    }
}
