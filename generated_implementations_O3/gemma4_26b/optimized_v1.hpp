#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* A_row = &A[i * K];
        float* C_row = &C[i * K];

        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            row_sum += A_row[p];
        }

        for (size_t j = 0; j < K; ++j) {
            C_row[j] = -row_sum;
        }

        for (size_t p = 0; p < K; ++p) {
            const float val = 2.0f * A_row[p];
            if (val == 0.0f) continue;

            const uint32_t* B_row_p = &B[p * K_ints];
            for (size_t chunk = 0; chunk < K_ints; ++chunk) {
                uint32_t word = B_row_p[chunk];
                while (word != 0) {
                    int bit_idx = __builtin_ctz(word);
                    C_row[chunk * 32 + bit_idx] += val;
                    word &= (word - 1);
                }
            }
        }
    }
}
