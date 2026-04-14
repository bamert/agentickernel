#pragma once

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_words = K >> 5;

    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;

        for (size_t j = 0; j < K; ++j) {
            const uint32_t* b_col = B + (j >> 5);
            const uint32_t mask = 1u << (j & 31);

            float sum = 0.0f;
            size_t p = 0;
            for (; p + 7 < K; p += 8) {
                const uint32_t b0 = b_col[(p + 0) * K_words];
                const uint32_t b1 = b_col[(p + 1) * K_words];
                const uint32_t b2 = b_col[(p + 2) * K_words];
                const uint32_t b3 = b_col[(p + 3) * K_words];
                const uint32_t b4 = b_col[(p + 4) * K_words];
                const uint32_t b5 = b_col[(p + 5) * K_words];
                const uint32_t b6 = b_col[(p + 6) * K_words];
                const uint32_t b7 = b_col[(p + 7) * K_words];

                sum += a_row[p + 0] * (float)((b0 & mask) ? 1 : -1);
                sum += a_row[p + 1] * (float)((b1 & mask) ? 1 : -1);
                sum += a_row[p + 2] * (float)((b2 & mask) ? 1 : -1);
                sum += a_row[p + 3] * (float)((b3 & mask) ? 1 : -1);
                sum += a_row[p + 4] * (float)((b4 & mask) ? 1 : -1);
                sum += a_row[p + 5] * (float)((b5 & mask) ? 1 : -1);
                sum += a_row[p + 6] * (float)((b6 & mask) ? 1 : -1);
                sum += a_row[p + 7] * (float)((b7 & mask) ? 1 : -1);
            }

            for (; p < K; ++p) {
                const uint32_t packed = b_col[p * K_words];
                sum += a_row[p] * (float)((packed & mask) ? 1 : -1);
            }

            c_row[j] = sum;
        }
    }
}
