#pragma once

void matmul(const float* A, const unsigned int* B, float* C, unsigned long M, unsigned long K) {
    unsigned long K_ints = K >> 5;
    for (unsigned long i = 0; i < M; ++i) {
        const float* arow = A + i * K;
        float* crow = C + i * K;
        for (unsigned long j = 0; j < K; ++j) crow[j] = 0.0f;

        for (unsigned long p = 0; p < K; ++p) {
            float aval = arow[p];
            const unsigned int* brow = B + p * K_ints;
            unsigned long base = 0;
            for (unsigned long w = 0; w < K_ints; ++w, base += 32) {
                unsigned int x = brow[w];
                crow[base + 0]  += aval * ((x & (1u << 0))  ? 1.0f : -1.0f);
                crow[base + 1]  += aval * ((x & (1u << 1))  ? 1.0f : -1.0f);
                crow[base + 2]  += aval * ((x & (1u << 2))  ? 1.0f : -1.0f);
                crow[base + 3]  += aval * ((x & (1u << 3))  ? 1.0f : -1.0f);
                crow[base + 4]  += aval * ((x & (1u << 4))  ? 1.0f : -1.0f);
                crow[base + 5]  += aval * ((x & (1u << 5))  ? 1.0f : -1.0f);
                crow[base + 6]  += aval * ((x & (1u << 6))  ? 1.0f : -1.0f);
                crow[base + 7]  += aval * ((x & (1u << 7))  ? 1.0f : -1.0f);
                crow[base + 8]  += aval * ((x & (1u << 8))  ? 1.0f : -1.0f);
                crow[base + 9]  += aval * ((x & (1u << 9))  ? 1.0f : -1.0f);
                crow[base + 10] += aval * ((x & (1u << 10)) ? 1.0f : -1.0f);
                crow[base + 11] += aval * ((x & (1u << 11)) ? 1.0f : -1.0f);
                crow[base + 12] += aval * ((x & (1u << 12)) ? 1.0f : -1.0f);
                crow[base + 13] += aval * ((x & (1u << 13)) ? 1.0f : -1.0f);
                crow[base + 14] += aval * ((x & (1u << 14)) ? 1.0f : -1.0f);
                crow[base + 15] += aval * ((x & (1u << 15)) ? 1.0f : -1.0f);
                crow[base + 16] += aval * ((x & (1u << 16)) ? 1.0f : -1.0f);
                crow[base + 17] += aval * ((x & (1u << 17)) ? 1.0f : -1.0f);
                crow[base + 18] += aval * ((x & (1u << 18)) ? 1.0f : -1.0f);
                crow[base + 19] += aval * ((x & (1u << 19)) ? 1.0f : -1.0f);
                crow[base + 20] += aval * ((x & (1u << 20)) ? 1.0f : -1.0f);
                crow[base + 21] += aval * ((x & (1u << 21)) ? 1.0f : -1.0f);
                crow[base + 22] += aval * ((x & (1u << 22)) ? 1.0f : -1.0f);
                crow[base + 23] += aval * ((x & (1u << 23)) ? 1.0f : -1.0f);
                crow[base + 24] += aval * ((x & (1u << 24)) ? 1.0f : -1.0f);
                crow[base + 25] += aval * ((x & (1u << 25)) ? 1.0f : -1.0f);
                crow[base + 26] += aval * ((x & (1u << 26)) ? 1.0f : -1.0f);
                crow[base + 27] += aval * ((x & (1u << 27)) ? 1.0f : -1.0f);
                crow[base + 28] += aval * ((x & (1u << 28)) ? 1.0f : -1.0f);
                crow[base + 29] += aval * ((x & (1u << 29)) ? 1.0f : -1.0f);
                crow[base + 30] += aval * ((x & (1u << 30)) ? 1.0f : -1.0f);
                crow[base + 31] += aval * ((x & (1u << 31)) ? 1.0f : -1.0f);
            }
        }
    }
}
