#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Initialize C with zeros
    for (size_t i = 0; i < M * K; ++i) {
        C[i] = 0.0f;
    }

    // Accumulate the dot product using a row-major-friendly traversal
    // We iterate through row A, then row B, to maximize cache locality for A and B.
    // However, since B is packed bits, B[p][j] needs to be expanded.
    
    for (size_t i = 0; i < M; ++i) {
        float* Ci = &C[i * K];
        const float* Ai = &A[i * K];
        
        for (size_t p = 0; p < K; ++p) {
            float a_val = Ai[p];
            const uint32_t* Bp = &B[p * K_ints];
            
            // For each integer in row p of B
            for (size_t kj = 0; kj < K_ints; ++kj) {
                uint32_t bits = Bp[kj];
                float* Cij_base = &Ci[kj * 32];
                
                // Process 32 bits at once
                for (int b = 0; b < 32; ++b) {
                    float sign = ((bits >> b) & 1) ? 1.0f : -1.0f;
                    Cij_base[b] += a_val * sign;
                }
            }
        }
    }
}
