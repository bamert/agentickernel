#pragma once
#include <vector>
#include <string>
#include <cstdint>

namespace baseline {
    #include "baseline.hpp"
}
namespace unroll4 {
    #include "unroll4.hpp"
}


namespace unroll8 {
    #include "unroll8.hpp"
}
namespace best {
    #include "best.hpp"
}
namespace best_scalar_unrolled {
    #include "best_scalar_unrolled.hpp"
}


// 2. Define the struct our main.cpp loop expects
struct KernelRegistry {
    std::string name;
    float (*fn)(const float*, const int8_t*, size_t);
};

// 3. Populate and return the registry
inline std::vector<KernelRegistry> all_match_kernels() {
    return {
    
        {"unroll4", unroll4::dot_product},
    
        {"baseline", baseline::dot_product},
    
        {"unroll8", unroll8::dot_product},
        {"best", best::dot_product},
        {"best_scalar_unrolled", best_scalar_unrolled::dot_product}
    
    };
}
