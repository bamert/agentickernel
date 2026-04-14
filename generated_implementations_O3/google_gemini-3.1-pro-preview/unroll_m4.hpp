#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; i += 4) {
        float row_sum[4] = {0.0f};
        for (size_t p = 0; p < K; ++p) {
            row_sum[0] += A[(i+0) * K + p];
            if (i+1 < M) row_sum[1] += A[(i+1) * K + p];
            if (i+2 < M) row_sum[2] += A[(i+2) * K + p];
            if (i+3 < M) row_sum[3] += A[(i+3) * K + p];
        }

        for (size_t j_word = 0; j_word < K_ints; ++j_word) {
            float sums[4][32] = {0};
            
            for (size_t p = 0; p < K; ++p) {
                float a0 = A[(i+0) * K + p];
                float a1 = (i+1 < M) ? A[(i+1) * K + p] : 0.0f;
                float a2 = (i+2 < M) ? A[(i+2) * K + p] : 0.0f;
                float a3 = (i+3 < M) ? A[(i+3) * K + p] : 0.0f;

                uint32_t packed = B[p * K_ints + j_word];
                
                for (int b = 0; b < 32; ++b) {
                    if ((packed >> b) & 1) {
                        sums[0][b] += a0;
                        sums[1][b] += a1;
                        sums[2][b] += a2;
                        sums[3][b] += a3;
                    }
                }
            }
            
            for (int b = 0; b < 32; ++b) {
                C[(i+0) * K + j_word * 32 + b] = sums[0][b] * 2.0f - row_sum[0];
                if (i+1 < M) C[(i+1) * K + j_word * 32 + b] = sums[1][b] * 2.0f - row_sum[1];
                if (i+2 < M) C[(i+2) * K + j_word * 32 + b] = sums[2][b] * 2.0f - row_sum[2];
                if (i+3 < M) C[(i+3) * K + j_word * 32 + b] = sums[3][b] * 2.0f - row_sum[3];
            }
        }
    }
}
