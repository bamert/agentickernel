#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>

inline float int_to_float_bitwise(uint32_t i) {
    float f;
    std::memcpy(&f, &i, sizeof(float));
    return f;
}

inline uint32_t float_to_int_bitwise(float f) {
    uint32_t i;
    std::memcpy(&i, &f, sizeof(float));
    return i;
}

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float sum_a = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            sum_a += A[i * K + p];
        }
        
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = -sum_a;
        }

        for (size_t p = 0; p < K; ++p) {
            float two_a = 2.0f * A[i * K + p];
            uint32_t two_a_i = float_to_int_bitwise(two_a);
            
            for (size_t j_blk = 0; j_blk < K_ints; ++j_blk) {
                uint32_t packed = B[p * K_ints + j_blk];
                size_t c_idx = i * K + j_blk * 32;
                
                for (int b = 0; b < 32; ++b) {
                    uint32_t bit = (packed >> b) & 1;
                    uint32_t mask = 0u - bit;
                    uint32_t add_val_i = two_a_i & mask;
                    C[c_idx + b] += int_to_float_bitwise(add_val_i);
                }
            }
        }
    }
}
