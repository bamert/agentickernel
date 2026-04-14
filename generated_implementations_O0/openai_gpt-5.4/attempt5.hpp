#pragma once

typedef unsigned int uint32_t;
typedef decltype(sizeof(0)) size_t;

template<int N>
inline void zero_block(float* p) {
    for (int i = 0; i < N; ++i) p[i] = 0.0f;
}

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;
    for (size_t i = 0; i < M; ++i) {
        const float* arow = A + i * K;
        float* crow = C + i * K;
        for (size_t j = 0; j < K; ++j) crow[j] = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            const float a = arow[p];
            const float twice = a + a;
            const uint32_t* brow = B + p * K_ints;
            size_t j = 0;
            for (size_t w = 0; w < K_ints; ++w, j += 32) {
                crow[j + 0] += -a; crow[j + 1] += -a; crow[j + 2] += -a; crow[j + 3] += -a;
                crow[j + 4] += -a; crow[j + 5] += -a; crow[j + 6] += -a; crow[j + 7] += -a;
                crow[j + 8] += -a; crow[j + 9] += -a; crow[j + 10] += -a; crow[j + 11] += -a;
                crow[j + 12] += -a; crow[j + 13] += -a; crow[j + 14] += -a; crow[j + 15] += -a;
                crow[j + 16] += -a; crow[j + 17] += -a; crow[j + 18] += -a; crow[j + 19] += -a;
                crow[j + 20] += -a; crow[j + 21] += -a; crow[j + 22] += -a; crow[j + 23] += -a;
                crow[j + 24] += -a; crow[j + 25] += -a; crow[j + 26] += -a; crow[j + 27] += -a;
                crow[j + 28] += -a; crow[j + 29] += -a; crow[j + 30] += -a; crow[j + 31] += -a;
                const uint32_t bits = brow[w];
                if (bits & (1u << 0)) crow[j + 0] += twice;
                if (bits & (1u << 1)) crow[j + 1] += twice;
                if (bits & (1u << 2)) crow[j + 2] += twice;
                if (bits & (1u << 3)) crow[j + 3] += twice;
                if (bits & (1u << 4)) crow[j + 4] += twice;
                if (bits & (1u << 5)) crow[j + 5] += twice;
                if (bits & (1u << 6)) crow[j + 6] += twice;
                if (bits & (1u << 7)) crow[j + 7] += twice;
                if (bits & (1u << 8)) crow[j + 8] += twice;
                if (bits & (1u << 9)) crow[j + 9] += twice;
                if (bits & (1u << 10)) crow[j + 10] += twice;
                if (bits & (1u << 11)) crow[j + 11] += twice;
                if (bits & (1u << 12)) crow[j + 12] += twice;
                if (bits & (1u << 13)) crow[j + 13] += twice;
                if (bits & (1u << 14)) crow[j + 14] += twice;
                if (bits & (1u << 15)) crow[j + 15] += twice;
                if (bits & (1u << 16)) crow[j + 16] += twice;
                if (bits & (1u << 17)) crow[j + 17] += twice;
                if (bits & (1u << 18)) crow[j + 18] += twice;
                if (bits & (1u << 19)) crow[j + 19] += twice;
                if (bits & (1u << 20)) crow[j + 20] += twice;
                if (bits & (1u << 21)) crow[j + 21] += twice;
                if (bits & (1u << 22)) crow[j + 22] += twice;
                if (bits & (1u << 23)) crow[j + 23] += twice;
                if (bits & (1u << 24)) crow[j + 24] += twice;
                if (bits & (1u << 25)) crow[j + 25] += twice;
                if (bits & (1u << 26)) crow[j + 26] += twice;
                if (bits & (1u << 27)) crow[j + 27] += twice;
                if (bits & (1u << 28)) crow[j + 28] += twice;
                if (bits & (1u << 29)) crow[j + 29] += twice;
                if (bits & (1u << 30)) crow[j + 30] += twice;
                if (bits & (1u << 31)) crow[j + 31] += twice;
            }
        }
    }
}
