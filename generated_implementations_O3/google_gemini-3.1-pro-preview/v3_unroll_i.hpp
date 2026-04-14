void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M * K; ++i) {
        C[i] = 0.0f;
    }

    size_t i = 0;
    for (; i + 3 < M; i += 4) {
        for (size_t p = 0; p < K; ++p) {
            float a0 = A[(i + 0) * K + p];
            float a1 = A[(i + 1) * K + p];
            float a2 = A[(i + 2) * K + p];
            float a3 = A[(i + 3) * K + p];

            const uint32_t* B_row = B + p * K_ints;
            float* C_row0 = C + (i + 0) * K;
            float* C_row1 = C + (i + 1) * K;
            float* C_row2 = C + (i + 2) * K;
            float* C_row3 = C + (i + 3) * K;

            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = B_row[j_int];
                for (size_t bit = 0; bit < 32; ++bit) {
                    size_t c_idx = j_int * 32 + bit;
                    if ((packed >> bit) & 1) {
                        C_row0[c_idx] += a0;
                        C_row1[c_idx] += a1;
                        C_row2[c_idx] += a2;
                        C_row3[c_idx] += a3;
                    } else {
                        C_row0[c_idx] -= a0;
                        C_row1[c_idx] -= a1;
                        C_row2[c_idx] -= a2;
                        C_row3[c_idx] -= a3;
                    }
                }
            }
        }
    }
    
    for (; i < M; ++i) {
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const uint32_t* B_row = B + p * K_ints;
            float* C_row = C + i * K;
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = B_row[j_int];
                for (size_t bit = 0; bit < 32; ++bit) {
                    size_t c_idx = j_int * 32 + bit;
                    if ((packed >> bit) & 1) {
                        C_row[c_idx] += a_val;
                    } else {
                        C_row[c_idx] -= a_val;
                    }
                }
            }
        }
    }
}
