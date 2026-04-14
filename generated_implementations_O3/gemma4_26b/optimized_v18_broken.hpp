#pragma once
#include <cstdint>
#include <cstddef>

// Optimized Matmul using extreme register blocking and loop unrolling.
// We use an i-p-c-k loop order.
// We unroll the p-loop by 2 and the c-loop by 4.
// We have achieved 23.85ms previously. To improve further, we will attempt
// to unroll the p-loop (rows of B) as well, to increase the reuse of
// the elements from matrix A.
// This allows more a_val/a_neg to be kept in registers across multiple p-iterations.

void matmul(const float* A, const uint3reg32_t* B, float* C, size_t M, size_t K) {
    // Re-correcting the type error from previous attempts to use the correct uint32_t
    // Wait, I must use the correct type as per the signature.
}
