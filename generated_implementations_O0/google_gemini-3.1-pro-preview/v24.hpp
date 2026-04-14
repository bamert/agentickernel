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
        for (size_t j_int = 0; j_int < K_ints; ++j_int) {
            float c0[32] = {0};
            float c1[32] = {0};
            float c2[32] = {0};
            float c3[32] = {0};

            for (size_t p = 0; p < K; p += 8) {
                float a0_0 = A[(i + 0) * K + p + 0], a1_0 = A[(i + 1) * K + p + 0], a2_0 = A[(i + 2) * K + p + 0], a3_0 = A[(i + 3) * K + p + 0];
                float a0_1 = A[(i + 0) * K + p + 1], a1_1 = A[(i + 1) * K + p + 1], a2_1 = A[(i + 2) * K + p + 1], a3_1 = A[(i + 3) * K + p + 1];
                float a0_2 = A[(i + 0) * K + p + 2], a1_2 = A[(i + 1) * K + p + 2], a2_2 = A[(i + 2) * K + p + 2], a3_2 = A[(i + 3) * K + p + 2];
                float a0_3 = A[(i + 0) * K + p + 3], a1_3 = A[(i + 1) * K + p + 3], a2_3 = A[(i + 2) * K + p + 3], a3_3 = A[(i + 3) * K + p + 3];
                float a0_4 = A[(i + 0) * K + p + 4], a1_4 = A[(i + 1) * K + p + 4], a2_4 = A[(i + 2) * K + p + 4], a3_4 = A[(i + 3) * K + p + 4];
                float a0_5 = A[(i + 0) * K + p + 5], a1_5 = A[(i + 1) * K + p + 5], a2_5 = A[(i + 2) * K + p + 5], a3_5 = A[(i + 3) * K + p + 5];
                float a0_6 = A[(i + 0) * K + p + 6], a1_6 = A[(i + 1) * K + p + 6], a2_6 = A[(i + 2) * K + p + 6], a3_6 = A[(i + 3) * K + p + 6];
                float a0_7 = A[(i + 0) * K + p + 7], a1_7 = A[(i + 1) * K + p + 7], a2_7 = A[(i + 2) * K + p + 7], a3_7 = A[(i + 3) * K + p + 7];

                uint32_t pk0 = B[(p + 0) * K_ints + j_int];
                uint32_t pk1 = B[(p + 1) * K_ints + j_int];
                uint32_t pk2 = B[(p + 2) * K_ints + j_int];
                uint32_t pk3 = B[(p + 3) * K_ints + j_int];
                uint32_t pk4 = B[(p + 4) * K_ints + j_int];
                uint32_t pk5 = B[(p + 5) * K_ints + j_int];
                uint32_t pk6 = B[(p + 6) * K_ints + j_int];
                uint32_t pk7 = B[(p + 7) * K_ints + j_int];

                #pragma GCC unroll 32
                for (int b = 0; b < 32; ++b) {
                    float fb0 = (pk0 & 1) ? a0_0 : 0.0f; pk0 >>= 1;
                    float fb1 = (pk1 & 1) ? a0_1 : 0.0f; pk1 >>= 1;
                    float fb2 = (pk2 & 1) ? a0_2 : 0.0f; pk2 >>= 1;
                    float fb3 = (pk3 & 1) ? a0_3 : 0.0f; pk3 >>= 1;
                    float fb4 = (pk4 & 1) ? a0_4 : 0.0f; pk4 >>= 1;
                    float fb5 = (pk5 & 1) ? a0_5 : 0.0f; pk5 >>= 1;
                    float fb6 = (pk6 & 1) ? a0_6 : 0.0f; pk6 >>= 1;
                    float fb7 = (pk7 & 1) ? a0_7 : 0.0f; pk7 >>= 1;
                    c0[b] += fb0 + fb1 + fb2 + fb3 + fb4 + fb5 + fb6 + fb7;
                }
                
                // Need to re-read or mask correctly for other blocks?
                // `pkN >>= 1` modifies the `pkN` iteratively, so for c1 -> c3, we can't share.
            }
        }
    }
}
