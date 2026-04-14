#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* Ci = &C[i * K];
        const float* Ai = &A[i * K];
        
        // Initialize Row Ci to -sum(Ai) since 0 bit means -1.0f
        // Let sum = sum(a_val * sign) where sign is 1 if bit=1, else -1
        // sum = sum(a_val * (2*bit - 1)) = sum(2 * a_val * bit - a_val)
        // sum = 2 * sum(a_val * bit) - sum(a_val)
        
        float row_sum_A = 0.0f;
        for (size_t p = 0; p < K; ++p) row_sum_A += Ai[p];
        for (size_t j = 0; j < K; ++j) Ci[j] = -row_sum_A;

        for (size_t p = 0; p < K; ++p) {
            float two_a = Ai[p] * 2.0f;
            const uint32_t* Bp = &B[p * K_ints];
            
            for (size_t kj = 0; kj < K_ints; ++kj) {
                uint32_t bits = Bp[kj];
                float* Cij = &Ci[kj * 32];
                
                if (bits == 0) continue;
                if (bits == 0xFFFFFFFF) {
                    for (int b = 0; b < 32; ++b) Cij[b] += two_a;
                    continue;
                }

                for (int b = 0; b < 32; ++b) {
                    if ((bits >> b) & 1) {
                        Cij[b] += two_a;
                    }
                }
            }
        }
    }
}
