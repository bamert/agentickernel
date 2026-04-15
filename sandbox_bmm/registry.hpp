#pragma once
#include <vector>
#include <string>
#include <cstdint>
#if defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace  baseline  {
    #include "baseline.hpp"
}
namespace matmul_neon_col_buffer {
    #include "gemma4_31b/matmul_neon_col_buffer.hpp"
}

struct KernelRegistry {
    std::string name;
    void (*fn)(const float*, const uint32_t*, float*, size_t, size_t);
};

inline std::vector<KernelRegistry> all_match_kernels() {
    return {
        // Assumes every file defines a function literally named `matmul`
        {"baseline", baseline::matmul},
        {"matmul_neon_col_buffer", matmul_neon_col_buffer::matmul}
    };
}
