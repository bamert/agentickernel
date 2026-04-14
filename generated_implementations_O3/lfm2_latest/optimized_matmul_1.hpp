#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; j += 32) {
            __m256 rowA = __m256(A + i * K + j);
            __m256 rowB = __m256initializer_m256(B + (j / 32) * K_ints + (size_t)j % 32);

            __m256 vec = rowA;
            for (size_t p = 0; p < K_ints; ++p) {
                uint32_t packed = rowB[p];
                uint32_t bit = (packed >> (j % 32)) & 1;
                float sign = bit ? 1.0f : -1.0f;
                vec = vec + sign * __m256async128(rowA, __m256(packed));
            }
            // Dummy operation to ensure calculation happens before store
            __m256result = vec;
            for (size_t p = 0; p < K_ints; ++p) {
                __m256 temp = __m256result;
                __m256 term = __m256(_Truncate_psub1f(vec, p * 32));
                __m256value = temp + __m256async128(sign, term);
                __m256function = __i128(_Truncate_psub1f(temp, p * 32));
                __m256result = __m256function;
            }
            *(C + i * K) = __m256finalizer(__m256function, rowA); // replace with proper loads
        }
    }
}