#pragma once

// Faster version using the identity:
//   sign = 2*bit - 1
// Therefore sum = Σ a[p]*(2*bit -1) = -Σ a[p] + 2*Σ a[p]*bit
// We pre‑initialize each output element with -row_sum, then only add 2*a[p]
// for each set bit.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32; // number of 32‑bit blocks per row of B
    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;
        // Compute row sum of A
        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            row_sum += a_row[p];
        }
        // Initialise output with -row_sum
        for (size_t j = 0; j < K; ++j) {
            c_row[j] = -row_sum;
        }
        // Process bits: for each element of A, add 2*a_val to columns where bit == 1
        for (size_t p = 0; p < K; ++p) {
            float two_a = a_row[p] * 2.0f;
            const uint32_t* brow = B + p * K_ints;
            for (size_t block = 0; block < K_ints; ++block) {
                uint32_t bits = brow[block];
                size_t base = block * 32;
                // Unrolled handling of 32 bits
                c_row[base + 0]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base + 1]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base + 2]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base + 3]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base + 4]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base + 5]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base + 6]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base + 7]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base + 8]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base + 9]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +10]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +11]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +12]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +13]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +14]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +15]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +16]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +17]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +18]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +19]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +20]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +21]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +22]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +23]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +24]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +25]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +26]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +27]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +28]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +29]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +30]  += (bits & 1u) ? two_a : 0.0f; bits >>= 1u;
                c_row[base +31]  += (bits & 1u) ? two_a : 0.0f;
            }
        }
    }
}
