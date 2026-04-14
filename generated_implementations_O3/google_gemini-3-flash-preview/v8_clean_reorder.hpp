#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Zero C
    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    for (size_t i = 0; i < M; ++i) {
        float* Ci = &C[i * K];
        const float* Ai = &A[i * K];
        
        for (size_t p = 0; p < K; ++p) {
            const float a_val = Ai[p];
            const uint32_t* Bp = &B[p * K_ints];
            
            for (size_t kj = 0; kj < K_ints; ++kj) {
                const uint32_t bits = Bp[kj];
                float* Cij = &Ci[kj * 32];
                
                // Using a slightly more compiler-friendly expansion
                // that doesn't rely on ternary branching if possible
                for (int b = 0; b < 32; ++b) {
                    // This often compiles to a conditional select or efficient math
                    float s = (bits & (1u << b)) ? a_val : -a_val;
                    Cij[b] += s;
                }
            }
        }
    }
}
