#pragma once

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;

    for (size_t i = 0; i < M; ++i) {
        const float* arow = A + i * K;
        float* crow = C + i * K;

        for (size_t j = 0; j < K; ++j) {
            const size_t packed_idx = j >> 5;
            const uint32_t mask = 1u << (j & 31);
            float sum = 0.0f;

            for (size_t p = 0; p < K; ++p) {
                const uint32_t packed = B[p * K_ints + packed_idx];
                sum += arow[p] * ((packed & mask) ? 1.0f : -1.0f);
            }

            crow[j] = sum;
        }
    }
}
