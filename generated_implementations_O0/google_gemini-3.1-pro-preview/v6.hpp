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

    size_t i = 0;
    for (; i + 3 < M; i += 4) {
        float* C0 = &C[(i + 0) * K];
        float* C1 = &C[(i + 1) * K];
        float* C2 = &C[(i + 2) * K];
        float* C3 = &C[(i + 3) * K];
        
        for (size_t j = 0; j < K; ++j) {
            C0[j] = 0.0f;
            C1[j] = 0.0f;
            C2[j] = 0.0f;
            C3[j] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) {
            float a0 = A[(i + 0) * K + p];
            float a1 = A[(i + 1) * K + p];
            float a2 = A[(i + 2) * K + p];
            float a3 = A[(i + 3) * K + p];

            const uint32_t* B_row = &B[p * K_ints];

            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = B_row[j_int];
                size_t base = j_int * 32;
                
                #pragma GCC unroll 32
                for (int b = 0; b < 32; ++b) {
                    float fbit = (packed >> b) & 1;
                    C0[base + b] += fbit * a0;
                    C1[base + b] += fbit * a1;
                    C2[base + b] += fbit * a2;
                    C3[base + b] += fbit * a3;
                }
            }
        }
        
        float rs0 = row_sums[i + 0];
        float rs1 = row_sums[i + 1];
        float rs2 = row_sums[i + 2];
        float rs3 = row_sums[i + 3];
        for (size_t j = 0; j < K; ++j) {
            C0[j] = 2.0f * C0[j] - rs0;
            C1[j] = 2.0f * C1[j] - rs1;
            C2[j] = 2.0f * C2[j] - rs2;
            C3[j] = 2.0f * C3[j] - rs3;
        }
    }
    
    for (; i < M; ++i) {
        float* C0 = &C[i * K];
        for (size_t j = 0; j < K; ++j) C0[j] = 0.0f;
        
        for (size_t p = 0; p < K; ++p) {
            float a0 = A[i * K + p];
            const uint32_t* B_row = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = B_row[j_int];
                size_t base = j_int * 32;
                for (int b = 0; b < 32; ++b) {
                    float fbit = (packed >> b) & 1;
                    C0[base + b] += fbit * a0;
                }
            }
        }
        
        float rs0 = row_sums[i];
        for (size_t j = 0; j < K; ++j) {
            C0[j] = 2.0f * C0[j] - rs0;
        }
    }
}
