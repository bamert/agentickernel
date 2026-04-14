#pragma once
#include <cstdint>
#include <cstddef>

// Calculates Matrix C = Matrix A * Matrix B
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
// ---------------------------------------------------------
// Optimization Strategy:
// 1. Tiling/Chunking: Instead of calculating one C[i][j] element at a time,
//    we compute a horizontal chunk of 32 elements of C simultaneously. This
//    aligns with the 32-bit packing of matrix B.
// 2. Data Reuse: By computing 32 output values at once, we can reuse the
//    value fetched from matrix A (A[i][p]) 32 times.
// 3. Register Accumulation: A local array `sums[32]` is used to store the
//    intermediate dot product results. This array is small and can likely be
//    held in registers by the compiler, reducing memory traffic to C until
//    the final values are computed.
// 4. Improved B Access: While still not perfectly sequential, this approach
//    reads a full uint32_t from B and uses all 32 of its bits before moving on,
//    making better use of the data loaded into cache.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* A_row = A + i * K;
        float* C_row = C + i * K;

        // Iterate through columns of B/C in chunks of 32
        for (size_t j_chunk = 0; j_chunk < K_ints; ++j_chunk) {
            float sums[32] = {0.0f};

            // This is the dot product dimension
            for (size_t p = 0; p < K; ++p) {
                const float a_val = A_row[p];
                const uint32_t packed_b = B[p * K_ints + j_chunk];
                
                // Update 32 accumulators at once using the bits from packed_b
                for (int bit = 0; bit < 32; ++bit) {
                    if ((packed_b >> bit) & 1) {
                        sums[bit] += a_val;
                    } else {
                        sums[bit] -= a_val;
                    }
                }
            }

            // Write the final 32 sums to the C matrix
            for (int k = 0; k < 32; ++k) {
                C_row[j_chunk * 32 + k] = sums[k];
            }
        }
    }
}
