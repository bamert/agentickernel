#pragma once

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_words = K >> 5;

    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;

        for (size_t j = 0; j < K; ++j) {
            const uint32_t* b_col_ptr = B + (j >> 5);
            const uint32_t mask = 1u << (j & 31);

            float sum = 0.0f;
            const float* a_ptr = a_row;
            const uint32_t* b_ptr = b_col_ptr;

            for (size_t p = 0; p < K; ++p) {
                const float a = *a_ptr++;
                const uint32_t packed = *b_ptr;
                b_ptr += K_words;
                sum += a * ((packed & mask) ? 1.0f : -1.0f);
            }

            c_row[j] = sum;
        }
    }
}
