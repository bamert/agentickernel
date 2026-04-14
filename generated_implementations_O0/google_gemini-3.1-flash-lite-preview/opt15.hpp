#pragma once
#include <cstdint>
#include <cstddef>

// Further optimize Opt14: 
// The bottleneck is likely memory access or the dependency chain in the inner loop.
// Since we have multiple elements in the same row of C (C[j_int*32 + 0...31]),
// loop unrolling helped nicely. Let's see if we can reduce indexing overhead
// by maintaining a raw pointer for B and an offset for the row_C pointer.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Direct initialization
    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    for (size_t i = 0; i < M; ++i) {
        const float* row_A = &A[i * K];
        float* row_C = &C[i * K];

        for (size_t p = 0; p < K; ++p) {
            float val_a = row_A[p];
            const uint32_t* B_row_p = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = B_row_p[j_int];
                float* c_ptr = row_C + (j_int << 5); // j_int * 32
                
                #pragma unroll
                for (int b = 0; b < 32; ++b) {
                    // CSEL is likely used. Using 1.0f or -1.0f is better.
                    c_ptr[b] += ((packed >> b) & 1) ? val_a : -val_a;
                }
            }
        }
    }
}
