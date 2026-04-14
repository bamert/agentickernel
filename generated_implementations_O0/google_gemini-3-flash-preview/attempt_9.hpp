
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    // A is M x K
    // B is row-major storage of bits: uint32_t B[p * K_ints + j / 32]
    // C is M x K
    
    // We want to calculate C[i, j] = sum over p of A[i, p] * sign(B[p, j])
    // The previous Attempt 3/7 was faster because it processed 32 columns of B (j to j+31)
    // for a single p. This reused A[i, p] across 32 B-elements.
    
    for (size_t i = 0; i < M; ++i) {
        float* row_C = &C[i * K];
        const float* row_A = &A[i * K];

        for (size_t bj = 0; bj < K_ints; ++bj) {
            // Process 32 columns in this bj block
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
            float s8 = 0.0f, s9 = 0.0f, s10 = 0.0f, s11 = 0.0f, s12 = 0.0f, s13 = 0.0f, s14 = 0.0f, s15 = 0.0f;
            float s16 = 0.0f, s17 = 0.0f, s18 = 0.0f, s19 = 0.0f, s20 = 0.0f, s21 = 0.0f, s22 = 0.0f, s23 = 0.0f;
            float s24 = 0.0f, s25 = 0.0f, s26 = 0.0f, s27 = 0.0f, s28 = 0.0f, s29 = 0.0f, s30 = 0.0f, s31 = 0.0f;

            for (size_t p = 0; p < K; ++p) {
                float a = row_A[p];
                uint32_t b = B[p * K_ints + bj];

                // Attempt 3 was: sum += (bit ? a : -a)
                // Let's use NEON to do 4 floats at once.
                // But since we are on a single core and don't want to overcomplicate,
                // let's try to unroll P slightly or improve the sign selection.
                
                // Keep the logic of Attempt 3 but optimize the sign application
                s0  += (b & (1u << 0))  ? a : -a;
                s1  += (b & (1u << 1))  ? a : -a;
                s2  += (b & (1u << 2))  ? a : -a;
                s3  += (b & (1u << 3))  ? a : -a;
                s4  += (b & (1u << 4))  ? a : -a;
                s5  += (b & (1u << 5))  ? a : -a;
                s6  += (b & (1u << 6))  ? a : -a;
                s7  += (b & (1u << 7))  ? a : -a;
                s8  += (b & (1u << 8))  ? a : -a;
                s9  += (b & (1u << 9))  ? a : -a;
                s10 += (b & (1u << 10)) ? a : -a;
                s11 += (b & (1u << 11)) ? a : -a;
                s12 += (b & (1u << 12)) ? a : -a;
                s13 += (b & (1u << 13)) ? a : -a;
                s14 += (b & (1u << 14)) ? a : -a;
                s15 += (b & (1u << 15)) ? a : -a;
                s16 += (b & (1u << 16)) ? a : -a;
                s17 += (b & (1u << 17)) ? a : -a;
                s18 += (b & (1u << 18)) ? a : -a;
                s19 += (b & (1u << 19)) ? a : -a;
                s20 += (b & (1u << 20)) ? a : -a;
                s21 += (b & (1u << 21)) ? a : -a;
                s22 += (b & (1u << 22)) ? a : -a;
                s23 += (b & (1u << 23)) ? a : -a;
                s24 += (b & (1u << 24)) ? a : -a;
                s25 += (b & (1u << 25)) ? a : -a;
                s26 += (b & (1u << 26)) ? a : -a;
                s27 += (b & (1u << 27)) ? a : -a;
                s28 += (b & (1u << 28)) ? a : -a;
                s29 += (b & (1u << 29)) ? a : -a;
                s30 += (b & (1u << 30)) ? a : -a;
                s31 += (b & (1u << 31)) ? a : -a;
            }

            row_C[bj * 32 + 0] = s0;   row_C[bj * 32 + 1] = s1;
            row_C[bj * 32 + 2] = s2;   row_C[bj * 32 + 3] = s3;
            row_C[bj * 32 + 4] = s4;   row_C[bj * 32 + 5] = s5;
            row_C[bj * 32 + 6] = s6;   row_C[bj * 32 + 7] = s7;
            row_C[bj * 32 + 8] = s8;   row_C[bj * 32 + 9] = s9;
            row_C[bj * 32 + 10] = s10; row_C[bj * 32 + 11] = s11;
            row_C[bj * 32 + 12] = s12; row_C[bj * 32 + 13] = s13;
            row_C[bj * 32 + 14] = s14; row_C[bj * 32 + 15] = s15;
            row_C[bj * 32 + 16] = s16; row_C[bj * 32 + 17] = s17;
            row_C[bj * 32 + 18] = s18; row_C[bj * 32 + 19] = s19;
            row_C[bj * 32 + 20] = s20; row_C[bj * 32 + 21] = s21;
            row_C[bj * 32 + 22] = s22; row_C[bj * 32 + 23] = s23;
            row_C[bj * 32 + 24] = s24; row_C[bj * 32 + 25] = s25;
            row_C[bj * 32 + 26] = s26; row_C[bj * 32 + 27] = s27;
            row_C[bj * 32 + 28] = s28; row_C[bj * 32 + 29] = s29;
            row_C[bj * 32 + 30] = s30; row_C[bj * 32 + 31] = s31;
        }
    }
}
