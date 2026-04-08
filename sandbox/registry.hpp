#pragma once
#include <vector>
#include <string>
#include <cstdint>

// 1. Wrap every agent file in its own namespace to prevent ODR errors

namespace baseline {
    #include "baseline.hpp"
}

namespace optimized_v2 {
    #include "optimized_v2.hpp"
}

namespace optimized_v3 {
    #include "optimized_v3.hpp"
}

namespace optimized_v1 {
    #include "optimized_v1.hpp"
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
    
        {"optimized_v2", optimized_v2::dot_product},
    
        {"optimized_v3", optimized_v3::dot_product},
    
        {"optimized_v1", optimized_v1::dot_product}
    
    };
}