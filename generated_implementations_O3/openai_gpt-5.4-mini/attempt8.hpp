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
            for (; p + 3 < K; p += 4) {
                const uint32_t b0 = b_col[(p + 0) * K_words];
                const uint32_t b1 = b_col[(p + 1) * K_words];
                const uint32_t b2 = b_col[(p + 2) * K_words];
                const uint32_t b3 = b_col[(p + 3) * K_words];

                const float s0 = (b0 & mask) ? 1.0f : -1.0f;
                const float s1 = (b1 & mask) ? 1.0f : -1.0f;
                const float s2 = (b2 & mask) ? 1.0f : -1.0f;
                const float s3 = (b3 & mask) ? 1.0f : -1.0f;

                sum += a_row[p + 0] * s0 + a_row[p + 1] * s1 + a_row[p + 2] * s2 + a_row[p + 3] * s3;
            }
            for (; p < K; ++p) {
                const uint32_t packed = b_col[p * K_words];
                sum += a_row[p] * ((packed & mask) ? 1.0f : -1.0f);
            }

            c_row[j] = sum;
        }
    }
}
