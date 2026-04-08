#ifndef __kernelperf_baseline
#define __kernelperf_baseline
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>

// Calculates the dot product of an FP32 vector and an INT8 vector.
// Every element in `vec2` is strictly -1, 0, or 1.
// ---------------------------------------------------------
float baseline(const float* vec1, const int8_t* vec2, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += vec1[i] * vec2[i];
    }
    return sum;
}


#endif 
