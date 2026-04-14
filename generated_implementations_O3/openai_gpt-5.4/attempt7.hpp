#pragma once

void matmul(const float* A, const unsigned int* B, float* C, unsigned long M, unsigned long K) {
    unsigned long K_ints = K >> 5;
    for (unsigned long i = 0; i < M; ++i) {
        const float* arow = A + i * K;
        float* crow = C + i * K;
        for (unsigned long j = 0; j < K; ++j) crow[j] = 0.0f;

        unsigned long p = 0;
        for (; p + 7 < K; p += 8) {
            float a0 = arow[p + 0], a1 = arow[p + 1], a2 = arow[p + 2], a3 = arow[p + 3];
            float a4 = arow[p + 4], a5 = arow[p + 5], a6 = arow[p + 6], a7 = arow[p + 7];
            const unsigned int* b0 = B + (p + 0) * K_ints;
            const unsigned int* b1 = B + (p + 1) * K_ints;
            const unsigned int* b2 = B + (p + 2) * K_ints;
            const unsigned int* b3 = B + (p + 3) * K_ints;
            const unsigned int* b4 = B + (p + 4) * K_ints;
            const unsigned int* b5 = B + (p + 5) * K_ints;
            const unsigned int* b6 = B + (p + 6) * K_ints;
            const unsigned int* b7 = B + (p + 7) * K_ints;

            unsigned long base = 0;
            for (unsigned long w = 0; w < K_ints; ++w, base += 32) {
                unsigned int x0 = b0[w], x1 = b1[w], x2 = b2[w], x3 = b3[w];
                unsigned int x4 = b4[w], x5 = b5[w], x6 = b6[w], x7 = b7[w];
                float* c = crow + base;

                c[0]  += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[1]  += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[2]  += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[3]  += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[4]  += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[5]  += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[6]  += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[7]  += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[8]  += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[9]  += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[10] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[11] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[12] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[13] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[14] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[15] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[16] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[17] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[18] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[19] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[20] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[21] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[22] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[23] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[24] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[25] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[26] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[27] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[28] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[29] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[30] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7); x0 >>= 1; x1 >>= 1; x2 >>= 1; x3 >>= 1; x4 >>= 1; x5 >>= 1; x6 >>= 1; x7 >>= 1;
                c[31] += (x0 & 1u ? a0 : -a0) + (x1 & 1u ? a1 : -a1) + (x2 & 1u ? a2 : -a2) + (x3 & 1u ? a3 : -a3) + (x4 & 1u ? a4 : -a4) + (x5 & 1u ? a5 : -a5) + (x6 & 1u ? a6 : -a6) + (x7 & 1u ? a7 : -a7);
            }
        }

        for (; p < K; ++p) {
            float aval = arow[p];
            const unsigned int* brow = B + p * K_ints;
            unsigned long base = 0;
            for (unsigned long w = 0; w < K_ints; ++w, base += 32) {
                unsigned int x = brow[w];
                float* c = crow + base;
                c[0] += (x & 1u) ? aval : -aval; x >>= 1;
                c[1] += (x & 1u) ? aval : -aval; x >>= 1;
                c[2] += (x & 1u) ? aval : -aval; x >>= 1;
                c[3] += (x & 1u) ? aval : -aval; x >>= 1;
                c[4] += (x & 1u) ? aval : -aval; x >>= 1;
                c[5] += (x & 1u) ? aval : -aval; x >>= 1;
                c[6] += (x & 1u) ? aval : -aval; x >>= 1;
                c[7] += (x & 1u) ? aval : -aval; x >>= 1;
                c[8] += (x & 1u) ? aval : -aval; x >>= 1;
                c[9] += (x & 1u) ? aval : -aval; x >>= 1;
                c[10] += (x & 1u) ? aval : -aval; x >>= 1;
                c[11] += (x & 1u) ? aval : -aval; x >>= 1;
                c[12] += (x & 1u) ? aval : -aval; x >>= 1;
                c[13] += (x & 1u) ? aval : -aval; x >>= 1;
                c[14] += (x & 1u) ? aval : -aval; x >>= 1;
                c[15] += (x & 1u) ? aval : -aval; x >>= 1;
                c[16] += (x & 1u) ? aval : -aval; x >>= 1;
                c[17] += (x & 1u) ? aval : -aval; x >>= 1;
                c[18] += (x & 1u) ? aval : -aval; x >>= 1;
                c[19] += (x & 1u) ? aval : -aval; x >>= 1;
                c[20] += (x & 1u) ? aval : -aval; x >>= 1;
                c[21] += (x & 1u) ? aval : -aval; x >>= 1;
                c[22] += (x & 1u) ? aval : -aval; x >>= 1;
                c[23] += (x & 1u) ? aval : -aval; x >>= 1;
                c[24] += (x & 1u) ? aval : -aval; x >>= 1;
                c[25] += (x & 1u) ? aval : -aval; x >>= 1;
                c[26] += (x & 1u) ? aval : -aval; x >>= 1;
                c[27] += (x & 1u) ? aval : -aval; x >>= 1;
                c[28] += (x & 1u) ? aval : -aval; x >>= 1;
                c[29] += (x & 1u) ? aval : -aval; x >>= 1;
                c[30] += (x & 1u) ? aval : -aval; x >>= 1;
                c[31] += (x & 1u) ? aval : -aval;
            }
        }
    }
}
