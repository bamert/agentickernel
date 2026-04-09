#pragma once
#include <vector>
#include <string>
#include <cstdint>
#if defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

// 1. Wrap every agent file in its own namespace to prevent ODR errors

namespace baseline {
    #include "baseline.hpp"
}

namespace attempt_24 {
    #include "attempt_24.hpp"
}


// 2. Define the struct our main.cpp loop expects
struct KernelRegistry {
    std::string name;
    float (*fn)(const float*, const int8_t*, size_t);
};

// 3. Populate and return the registry
inline std::vector<KernelRegistry> all_match_kernels() {
    return {
    
        {"baseline", baseline::dot_product},
    
        {"attempt_24", attempt_24::dot_product}
    
    };
}