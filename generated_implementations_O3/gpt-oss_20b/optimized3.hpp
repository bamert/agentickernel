#pragma once
#include <cstdint>
#include <cstddef>

// Optimized matmul with improved inner-loop: eliminate modulo/divide by 32
// and unroll across 8 columns.  The memory layout is still row-major for A and C
// and packed binary for B.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    // Zero C.
    for (size_t idx = 0; idx < M * K; ++idx) {
        C[idx] = 0.0f;
    }

    for (size_t i = 0; i < M; ++i) {
        float* c_row = &C[i * K];
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const uint32_t* b_row = &B[p * K_ints];

            uint32_t packed = *b_row++;
            unsigned shift = 0;
            size_t j = 0;
            // Unroll loop 8 times
            for (; j + 7 < K; j += 8) {
                unsigned bit0 = (packed >> shift) & 1U; shift++; if (shift == 32) { packed = *b_row++; shift = 0; }
                unsigned bit1 = (packed >> shift) & 1U; shift++; if (shift == 32) { packed = *b_row++; shift = 0; }
                unsigned bit2 = (packed >> shift) & 1U; shift++; if (shift == 32) { packed = *b_row++; shift = 0; }
                unsigned bit3 = (packed >> shift) & 1U; shift++; if (shift == 32) { packed = *b_row++; shift = 0; }
                unsigned bit4 = (packed >> shift) & 1U; shift++; if (shift == 32) { packed = *b_row++; shift = 0; }
                unsigned bit5 = (packed >> shift) & 1U; shift++; if (shift == 32) { packed = *b_row++; shift = 0; }
                unsigned bit6 = (packed >> shift) & 1U; shift++; if (shift == 32) { packed = *b_row++; shift = 0; }
                unsigned bit7 = (packed >> shift) & 1U; shift++; if (shift == 32) { packed = *b_row++; shift = 0; }

                c_row[j]     += a_val * (bit0 ? 1.0f : -1.0f);
                c_row[j + 1] += a_val * (bit1 ? 1.0f : -1.0f);
                c_row[j + 2] += a_val * (bit2 ? 1.0f : -1.0f);
                c_row[j + 3] += a_val * (bit3 ? 1.0f : -1.0f);
                c_row[j + 4] += a_val * (bit4 ? 1.0f : -1.0f);
                c_row[j + 5] += a_val * (bit5 ? 1.0f : -1.0f);
                c_row[j + 6] += a_val * (bit6 ? 1.0f : -1.0f);
                c_row[j + 7] += a_val * (bit7 ? 1.0f : -1.0f);
            }
            // Handle tail columns
            for (; j < K; ++j) {
                unsigned bit = (packed >> shift) & 1U; shift++; if (shift == 32) { packed = *b_row++; shift = 0; }
                c_row[j] += a_val * (bit ? 1.0f : -1.0f);
            }
        }
    }
}
