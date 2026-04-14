#pragma once

void matmul(const float* A, const unsigned int* B, float* C, unsigned long M, unsigned long K) {
    unsigned long K_ints = K >> 5;
    for (unsigned long i = 0; i < M; ++i) {
        const float* arow = A + i * K;
        float* crow = C + i * K;
        for (unsigned long j = 0; j < K; ++j) crow[j] = 0.0f;

        unsigned long p = 0;
        for (; p + 3 < K; p += 4) {
            float a0 = arow[p + 0];
            float a1 = arow[p + 1];
            float a2 = arow[p + 2];
            float a3 = arow[p + 3];
            const unsigned int* b0 = B + (p + 0) * K_ints;
            const unsigned int* b1 = B + (p + 1) * K_ints;
            const unsigned int* b2 = B + (p + 2) * K_ints;
            const unsigned int* b3 = B + (p + 3) * K_ints;

            unsigned long base = 0;
            for (unsigned long w = 0; w < K_ints; ++w, base += 32) {
                unsigned int x0 = b0[w];
                unsigned int x1 = b1[w];
                unsigned int x2 = b2[w];
                unsigned int x3 = b3[w];

                crow[base + 0]  += ((x0 & (1u << 0))  ? a0 : -a0) + ((x1 & (1u << 0))  ? a1 : -a1) + ((x2 & (1u << 0))  ? a2 : -a2) + ((x3 & (1u << 0))  ? a3 : -a3);
                crow[base + 1]  += ((x0 & (1u << 1))  ? a0 : -a0) + ((x1 & (1u << 1))  ? a1 : -a1) + ((x2 & (1u << 1))  ? a2 : -a2) + ((x3 & (1u << 1))  ? a3 : -a3);
                crow[base + 2]  += ((x0 & (1u << 2))  ? a0 : -a0) + ((x1 & (1u << 2))  ? a1 : -a1) + ((x2 & (1u << 2))  ? a2 : -a2) + ((x3 & (1u << 2))  ? a3 : -a3);
                crow[base + 3]  += ((x0 & (1u << 3))  ? a0 : -a0) + ((x1 & (1u << 3))  ? a1 : -a1) + ((x2 & (1u << 3))  ? a2 : -a2) + ((x3 & (1u << 3))  ? a3 : -a3);
                crow[base + 4]  += ((x0 & (1u << 4))  ? a0 : -a0) + ((x1 & (1u << 4))  ? a1 : -a1) + ((x2 & (1u << 4))  ? a2 : -a2) + ((x3 & (1u << 4))  ? a3 : -a3);
                crow[base + 5]  += ((x0 & (1u << 5))  ? a0 : -a0) + ((x1 & (1u << 5))  ? a1 : -a1) + ((x2 & (1u << 5))  ? a2 : -a2) + ((x3 & (1u << 5))  ? a3 : -a3);
                crow[base + 6]  += ((x0 & (1u << 6))  ? a0 : -a0) + ((x1 & (1u << 6))  ? a1 : -a1) + ((x2 & (1u << 6))  ? a2 : -a2) + ((x3 & (1u << 6))  ? a3 : -a3);
                crow[base + 7]  += ((x0 & (1u << 7))  ? a0 : -a0) + ((x1 & (1u << 7))  ? a1 : -a1) + ((x2 & (1u << 7))  ? a2 : -a2) + ((x3 & (1u << 7))  ? a3 : -a3);
                crow[base + 8]  += ((x0 & (1u << 8))  ? a0 : -a0) + ((x1 & (1u << 8))  ? a1 : -a1) + ((x2 & (1u << 8))  ? a2 : -a2) + ((x3 & (1u << 8))  ? a3 : -a3);
                crow[base + 9]  += ((x0 & (1u << 9))  ? a0 : -a0) + ((x1 & (1u << 9))  ? a1 : -a1) + ((x2 & (1u << 9))  ? a2 : -a2) + ((x3 & (1u << 9))  ? a3 : -a3);
                crow[base + 10] += ((x0 & (1u << 10)) ? a0 : -a0) + ((x1 & (1u << 10)) ? a1 : -a1) + ((x2 & (1u << 10)) ? a2 : -a2) + ((x3 & (1u << 10)) ? a3 : -a3);
                crow[base + 11] += ((x0 & (1u << 11)) ? a0 : -a0) + ((x1 & (1u << 11)) ? a1 : -a1) + ((x2 & (1u << 11)) ? a2 : -a2) + ((x3 & (1u << 11)) ? a3 : -a3);
                crow[base + 12] += ((x0 & (1u << 12)) ? a0 : -a0) + ((x1 & (1u << 12)) ? a1 : -a1) + ((x2 & (1u << 12)) ? a2 : -a2) + ((x3 & (1u << 12)) ? a3 : -a3);
                crow[base + 13] += ((x0 & (1u << 13)) ? a0 : -a0) + ((x1 & (1u << 13)) ? a1 : -a1) + ((x2 & (1u << 13)) ? a2 : -a2) + ((x3 & (1u << 13)) ? a3 : -a3);
                crow[base + 14] += ((x0 & (1u << 14)) ? a0 : -a0) + ((x1 & (1u << 14)) ? a1 : -a1) + ((x2 & (1u << 14)) ? a2 : -a2) + ((x3 & (1u << 14)) ? a3 : -a3);
                crow[base + 15] += ((x0 & (1u << 15)) ? a0 : -a0) + ((x1 & (1u << 15)) ? a1 : -a1) + ((x2 & (1u << 15)) ? a2 : -a2) + ((x3 & (1u << 15)) ? a3 : -a3);
                crow[base + 16] += ((x0 & (1u << 16)) ? a0 : -a0) + ((x1 & (1u << 16)) ? a1 : -a1) + ((x2 & (1u << 16)) ? a2 : -a2) + ((x3 & (1u << 16)) ? a3 : -a3);
                crow[base + 17] += ((x0 & (1u << 17)) ? a0 : -a0) + ((x1 & (1u << 17)) ? a1 : -a1) + ((x2 & (1u << 17)) ? a2 : -a2) + ((x3 & (1u << 17)) ? a3 : -a3);
                crow[base + 18] += ((x0 & (1u << 18)) ? a0 : -a0) + ((x1 & (1u << 18)) ? a1 : -a1) + ((x2 & (1u << 18)) ? a2 : -a2) + ((x3 & (1u << 18)) ? a3 : -a3);
                crow[base + 19] += ((x0 & (1u << 19)) ? a0 : -a0) + ((x1 & (1u << 19)) ? a1 : -a1) + ((x2 & (1u << 19)) ? a2 : -a2) + ((x3 & (1u << 19)) ? a3 : -a3);
                crow[base + 20] += ((x0 & (1u << 20)) ? a0 : -a0) + ((x1 & (1u << 20)) ? a1 : -a1) + ((x2 & (1u << 20)) ? a2 : -a2) + ((x3 & (1u << 20)) ? a3 : -a3);
                crow[base + 21] += ((x0 & (1u << 21)) ? a0 : -a0) + ((x1 & (1u << 21)) ? a1 : -a1) + ((x2 & (1u << 21)) ? a2 : -a2) + ((x3 & (1u << 21)) ? a3 : -a3);
                crow[base + 22] += ((x0 & (1u << 22)) ? a0 : -a0) + ((x1 & (1u << 22)) ? a1 : -a1) + ((x2 & (1u << 22)) ? a2 : -a2) + ((x3 & (1u << 22)) ? a3 : -a3);
                crow[base + 23] += ((x0 & (1u << 23)) ? a0 : -a0) + ((x1 & (1u << 23)) ? a1 : -a1) + ((x2 & (1u << 23)) ? a2 : -a2) + ((x3 & (1u << 23)) ? a3 : -a3);
                crow[base + 24] += ((x0 & (1u << 24)) ? a0 : -a0) + ((x1 & (1u << 24)) ? a1 : -a1) + ((x2 & (1u << 24)) ? a2 : -a2) + ((x3 & (1u << 24)) ? a3 : -a3);
                crow[base + 25] += ((x0 & (1u << 25)) ? a0 : -a0) + ((x1 & (1u << 25)) ? a1 : -a1) + ((x2 & (1u << 25)) ? a2 : -a2) + ((x3 & (1u << 25)) ? a3 : -a3);
                crow[base + 26] += ((x0 & (1u << 26)) ? a0 : -a0) + ((x1 & (1u << 26)) ? a1 : -a1) + ((x2 & (1u << 26)) ? a2 : -a2) + ((x3 & (1u << 26)) ? a3 : -a3);
                crow[base + 27] += ((x0 & (1u << 27)) ? a0 : -a0) + ((x1 & (1u << 27)) ? a1 : -a1) + ((x2 & (1u << 27)) ? a2 : -a2) + ((x3 & (1u << 27)) ? a3 : -a3);
                crow[base + 28] += ((x0 & (1u << 28)) ? a0 : -a0) + ((x1 & (1u << 28)) ? a1 : -a1) + ((x2 & (1u << 28)) ? a2 : -a2) + ((x3 & (1u << 28)) ? a3 : -a3);
                crow[base + 29] += ((x0 & (1u << 29)) ? a0 : -a0) + ((x1 & (1u << 29)) ? a1 : -a1) + ((x2 & (1u << 29)) ? a2 : -a2) + ((x3 & (1u << 29)) ? a3 : -a3);
                crow[base + 30] += ((x0 & (1u << 30)) ? a0 : -a0) + ((x1 & (1u << 30)) ? a1 : -a1) + ((x2 & (1u << 30)) ? a2 : -a2) + ((x3 & (1u << 30)) ? a3 : -a3);
                crow[base + 31] += ((x0 & (1u << 31)) ? a0 : -a0) + ((x1 & (1u << 31)) ? a1 : -a1) + ((x2 & (1u << 31)) ? a2 : -a2) + ((x3 & (1u << 31)) ? a3 : -a3);
            }
        }

        for (; p < K; ++p) {
            float aval = arow[p];
            const unsigned int* brow = B + p * K_ints;
            unsigned long base = 0;
            for (unsigned long w = 0; w < K_ints; ++w, base += 32) {
                unsigned int x = brow[w];
                crow[base + 0]  += (x & (1u << 0))  ? aval : -aval;
                crow[base + 1]  += (x & (1u << 1))  ? aval : -aval;
                crow[base + 2]  += (x & (1u << 2))  ? aval : -aval;
                crow[base + 3]  += (x & (1u << 3))  ? aval : -aval;
                crow[base + 4]  += (x & (1u << 4))  ? aval : -aval;
                crow[base + 5]  += (x & (1u << 5))  ? aval : -aval;
                crow[base + 6]  += (x & (1u << 6))  ? aval : -aval;
                crow[base + 7]  += (x & (1u << 7))  ? aval : -aval;
                crow[base + 8]  += (x & (1u << 8))  ? aval : -aval;
                crow[base + 9]  += (x & (1u << 9))  ? aval : -aval;
                crow[base + 10] += (x & (1u << 10)) ? aval : -aval;
                crow[base + 11] += (x & (1u << 11)) ? aval : -aval;
                crow[base + 12] += (x & (1u << 12)) ? aval : -aval;
                crow[base + 13] += (x & (1u << 13)) ? aval : -aval;
                crow[base + 14] += (x & (1u << 14)) ? aval : -aval;
                crow[base + 15] += (x & (1u << 15)) ? aval : -aval;
                crow[base + 16] += (x & (1u << 16)) ? aval : -aval;
                crow[base + 17] += (x & (1u << 17)) ? aval : -aval;
                crow[base + 18] += (x & (1u << 18)) ? aval : -aval;
                crow[base + 19] += (x & (1u << 19)) ? aval : -aval;
                crow[base + 20] += (x & (1u << 20)) ? aval : -aval;
                crow[base + 21] += (x & (1u << 21)) ? aval : -aval;
                crow[base + 22] += (x & (1u << 22)) ? aval : -aval;
                crow[base + 23] += (x & (1u << 23)) ? aval : -aval;
                crow[base + 24] += (x & (1u << 24)) ? aval : -aval;
                crow[base + 25] += (x & (1u << 25)) ? aval : -aval;
                crow[base + 26] += (x & (1u << 26)) ? aval : -aval;
                crow[base + 27] += (x & (1u << 27)) ? aval : -aval;
                crow[base + 28] += (x & (1u << 28)) ? aval : -aval;
                crow[base + 29] += (x & (1u << 29)) ? aval : -aval;
                crow[base + 30] += (x & (1u << 30)) ? aval : -aval;
                crow[base + 31] += (x & (1u << 31)) ? aval : -aval;
            }
        }
    }
}
