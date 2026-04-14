#pragma once

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_words = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;

        for (size_t j = 0; j < K; ++j) {
            const size_t word_idx = j >> 5;
            const uint32_t mask = 1u << (j & 31);
            const uint32_t* b_row = B;

            float sum = 0.0f;
            for (size_t p = 0; p < K; ++p) {
                const uint32_t packed = b_row[p * K_words + word_idx];
                const float sign = (packed & mask) ? 1.0f : -1.0f;
                sum += a_row[p] * sign;
            }
            c_row[j] = sum;
        }
    }
}
