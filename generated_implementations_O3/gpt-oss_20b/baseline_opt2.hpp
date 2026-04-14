#pragma once
#include <cstdint>
#include <cstddef>

// Baseline Optimized 2: Slight perf tweaks – branchless sign calculation, no temporary sign vector.
// Still scalar, but removes a ternary per bit and replaces it with a small arithmetic expression.
// Assumes K is a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;
        // Zero output row
        for (size_t j = 0; j < K; ++j) c_row[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            size_t base = p * K_ints;
            for (size_t wi = 0; wi < K_ints; ++wi) {
                uint32_t packed = B[base + wi];
                for (size_t t = 0; t < 32; ++t) {
                    size_t j = wi * 32 + t;
                    // Branchless computation of sign * a_val
                    float bitf = (float)((packed >> t) & 1U);
                    c_row[j] += (bitf * 2.0f - 1.0f) * a_val;
                }
            }
        }
    }
}
