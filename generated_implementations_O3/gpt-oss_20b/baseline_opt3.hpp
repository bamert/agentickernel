#pragma once
#include <cstdint>
#include <cstddef>

// Baseline Optimized 3: Fully unroll 32-bit inner loop.
// This removes the inner loop overhead, which can give a modest
// performance improvement on modern CPUs.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;
        for (size_t j = 0; j < K; ++j) c_row[j] = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            size_t base = p * K_ints;
            for (size_t wi = 0; wi < K_ints; ++wi) {
                uint32_t packed = B[base + wi];
                // Manually unroll 32 iterations
                size_t j = wi * 32 + 0;  float sign0 = ((packed >> 0) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign0;
                j = wi * 32 + 1;  float sign1 = ((packed >> 1) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign1;
                j = wi * 32 + 2;  float sign2 = ((packed >> 2) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign2;
                j = wi * 32 + 3;  float sign3 = ((packed >> 3) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign3;
                j = wi * 32 + 4;  float sign4 = ((packed >> 4) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign4;
                j = wi * 32 + 5;  float sign5 = ((packed >> 5) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign5;
                j = wi * 32 + 6;  float sign6 = ((packed >> 6) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign6;
                j = wi * 32 + 7;  float sign7 = ((packed >> 7) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign7;
                j = wi * 32 + 8;  float sign8 = ((packed >> 8) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign8;
                j = wi * 32 + 9;  float sign9 = ((packed >> 9) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign9;
                j = wi * 32 + 10; float sign10 = ((packed >> 10) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign10;
                j = wi * 32 + 11; float sign11 = ((packed >> 11) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign11;
                j = wi * 32 + 12; float sign12 = ((packed >> 12) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign12;
                j = wi * 32 + 13; float sign13 = ((packed >> 13) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign13;
                j = wi * 32 + 14; float sign14 = ((packed >> 14) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign14;
                j = wi * 32 + 15; float sign15 = ((packed >> 15) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign15;
                j = wi * 32 + 16; float sign16 = ((packed >> 16) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign16;
                j = wi * 32 + 17; float sign17 = ((packed >> 17) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign17;
                j = wi * 32 + 18; float sign18 = ((packed >> 18) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign18;
                j = wi * 32 + 19; float sign19 = ((packed >> 19) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign19;
                j = wi * 32 + 20; float sign20 = ((packed >> 20) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign20;
                j = wi * 32 + 21; float sign21 = ((packed >> 21) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign21;
                j = wi * 32 + 22; float sign22 = ((packed >> 22) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign22;
                j = wi * 32 + 23; float sign23 = ((packed >> 23) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign23;
                j = wi * 32 + 24; float sign24 = ((packed >> 24) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign24;
                j = wi * 32 + 25; float sign25 = ((packed >> 25) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign25;
                j = wi * 32 + 26; float sign26 = ((packed >> 26) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign26;
                j = wi * 32 + 27; float sign27 = ((packed >> 27) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign27;
                j = wi * 32 + 28; float sign28 = ((packed >> 28) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign28;
                j = wi * 32 + 29; float sign29 = ((packed >> 29) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign29;
                j = wi * 32 + 30; float sign30 = ((packed >> 30) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign30;
                j = wi * 32 + 31; float sign31 = ((packed >> 31) & 1U) ? 1.0f : -1.0f; c_row[j] += a_val * sign31;
            }
        }
    }
}
