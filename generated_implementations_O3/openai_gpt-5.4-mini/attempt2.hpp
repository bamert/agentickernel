#pragma once

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_words = K >> 5;

    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;

        for (size_t j = 0; j < K; ++j) {
            const size_t word_idx = j >> 5;
            const uint32_t mask = 1u << (j & 31);

            const uint32_t* b_ptr = B + word_idx;
            const float* a_ptr = a_row;
            size_t p = 0;
            float sum = 0.0f;

            for (; p + 3 < K; p += 4) {
                const uint32_t b0 = b_ptr[(p + 0) * K_words];
                const uint32_t b1 = b_ptr[(p + 1) * K_words];
                const uint32_t b2 = b_ptr[(p + 2) * K_words];
                const uint32_t b3 = b_ptr[(p + 3) * K_words];

                sum += a_ptr[0] * ((b0 & mask) ? 1.0f : -1.0f);
                sum += a_ptr[1] * ((b1 & mask) ? 1.0f : -1.0f);
                sum += a_ptr[2] * ((b2 & mask) ? 1.0f : -1.0f);
                sum += a_ptr[3] * ((b3 & mask) ? 1.0f : -1.0f);

                a_ptr += 4;
            }

            for (; p < K; ++p) {
                const uint32_t packed = b_ptr[p * K_words];
                sum += *a_ptr * ((packed & mask) ? 1.0f : -1.0f);
                ++a_ptr;
            }

            c_row[j] = sum;
        }
    }
}
