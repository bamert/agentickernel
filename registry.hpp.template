#ifndef __kernelperf_registry
#define __kernelperf_registry
#include <vector>
#include <cstdint>
#include <string>
#include "baseline.hpp"
using KernelSignature = float (*)(const float* vec1, const int8_t* vec2, size_t n);

struct Kernel {
    std::string name;
    KernelSignature fn;
};
inline void PrintTo(const Kernel& k, std::ostream* os) {
    *os << k.name;
}
std::vector<Kernel> all_match_kernels() {
    return {
        {"baseline", &baseline},
};
}
#endif
