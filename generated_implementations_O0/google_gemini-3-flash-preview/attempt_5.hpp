
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* row_C = &C[i * K];
        const float* row_A = &A[i * K];
        
        for (size_t bj = 0; bj < K_ints; ++bj) {
            float sums0 = 0, sums1 = 0, sums2 = 0, sums3 = 0, sums4 = 0, sums5 = 0, sums6 = 0, sums7 = 0;
            float sums8 = 0, sums9 = 0, sums10 = 0, sums11 = 0, sums12 = 0, sums13 = 0, sums14 = 0, sums15 = 0;
            float sums16 = 0, sums17 = 0, sums18 = 0, sums19 = 0, sums20 = 0, sums21 = 0, sums22 = 0, sums23 = 0;
            float sums24 = 0, sums25 = 0, sums26 = 0, sums27 = 0, sums28 = 0, sums29 = 0, sums30 = 0, sums31 = 0;
            
            for (size_t p = 0; p < K; ++p) {
                float aval = row_A[p];
                uint32_t bval = B[p * K_ints + bj];
                
                sums0  += (bval & (1u << 0))  ? aval : -aval;
                sums1  += (bval & (1u << 1))  ? aval : -aval;
                sums2  += (bval & (1u << 2))  ? aval : -aval;
                sums3  += (bval & (1u << 3))  ? aval : -aval;
                sums4  += (bval & (1u << 4))  ? aval : -aval;
                sums5  += (bval & (1u << 5))  ? aval : -aval;
                sums6  += (bval & (1u << 6))  ? aval : -aval;
                sums7  += (bval & (1u << 7))  ? aval : -aval;
                sums8  += (bval & (1u << 8))  ? aval : -aval;
                sums9  += (bval & (1u << 9))  ? aval : -aval;
                sums10 += (bval & (1u << 10)) ? aval : -aval;
                sums11 += (bval & (1u << 11)) ? aval : -aval;
                sums12 += (bval & (1u << 12)) ? aval : -aval;
                sums13 += (bval & (1u << 13)) ? aval : -aval;
                sums14 += (bval & (1u << 14)) ? aval : -aval;
                sums15 += (bval & (1u << 15)) ? aval : -aval;
                sums16 += (bval & (1u << 16)) ? aval : -aval;
                sums17 += (bval & (1u << 17)) ? aval : -aval;
                sums18 += (bval & (1u << 18)) ? aval : -aval;
                sums19 += (bval & (1u << 19)) ? aval : -aval;
                sums20 += (bval & (1u << 20)) ? aval : -aval;
                sums21 += (bval & (1u << 21)) ? aval : -aval;
                sums22 += (bval & (1u << 22)) ? aval : -aval;
                sums23 += (bval & (1u << 23)) ? aval : -aval;
                sums24 += (bval & (1u << 24)) ? aval : -aval;
                sums25 += (bval & (1u << 25)) ? aval : -aval;
                sums26 += (bval & (1u << 26)) ? aval : -aval;
                sums27 += (bval & (1u << 27)) ? aval : -aval;
                sums28 += (bval & (1u << 28)) ? aval : -aval;
                sums29 += (bval & (1u << 29)) ? aval : -aval;
                sums30 += (bval & (1u << 30)) ? aval : -aval;
                sums31 += (bval & (1u << 31)) ? aval : -aval;
            }
            
            row_C[bj * 32 + 0] = sums0;
            row_C[bj * 32 + 1] = sums1;
            row_C[bj * 32 + 2] = sums2;
            row_C[bj * 32 + 3] = sums3;
            row_C[bj * 32 + 4] = sums4;
            row_C[bj * 32 + 5] = sums5;
            row_C[bj * 32 + 6] = sums6;
            row_C[bj * 32 + 7] = sums7;
            row_C[bj * 32 + 8] = sums8;
            row_C[bj * 32 + 9] = sums9;
            row_C[bj * 32 + 10] = sums10;
            row_C[bj * 32 + 11] = sums11;
            row_C[bj * 32 + 12] = sums12;
            row_C[bj * 32 + 13] = sums13;
            row_C[bj * 32 + 14] = sums14;
            row_C[bj * 32 + 15] = sums15;
            row_C[bj * 32 + 16] = sums16;
            row_C[bj * 32 + 17] = sums17;
            row_C[bj * 32 + 18] = sums18;
            row_C[bj * 32 + 19] = sums19;
            row_C[bj * 32 + 20] = sums20;
            row_C[bj * 32 + 21] = sums21;
            row_C[bj * 32 + 22] = sums22;
            row_C[bj * 32 + 23] = sums23;
            row_C[bj * 32 + 24] = sums24;
            row_C[bj * 32 + 25] = sums25;
            row_C[bj * 32 + 26] = sums26;
            row_C[bj * 32 + 27] = sums27;
            row_C[bj * 32 + 28] = sums28;
            row_C[bj * 32 + 29] = sums29;
            row_C[bj * 32 + 30] = sums30;
            row_C[bj * 32 + 31] = sums31;
        }
    }
}
