#pragma once
#include <vector>
#include <string>
#include <cstdint>
#if defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

// 1. Wrap every agent file in its own namespace to prevent ODR errors
namespace  baseline  {
    #include "baseline.hpp"
}
namespace optimized22 {
    #include "gpt-oss_120b/optimized22.hpp"
}

// 2. Define the struct our main.cpp loop expects
struct KernelRegistry {
    std::string name;
    // UPDATED SIGNATURE: Matrix A, Matrix B, Matrix C, M (rows), K (inner dim)
    void (*fn)(const float*, const uint32_t*, float*, size_t, size_t);
};

// 3. Populate and return the registry
inline std::vector<KernelRegistry> all_match_kernels() {
    return {
        // Assumes every file defines a function literally named `matmul`
        {"baseline", baseline::matmul},
        {"optimized22", optimized22::matmul}
    };
}
