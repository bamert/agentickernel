#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <random>
#include "registry.hpp"
#include <benchmark/benchmark.h>

inline float generate_data(size_t N, std::vector<float>& vec1, std::vector<int8_t>& vec2) {
    vec1.resize(N);
    vec2.resize(N);
    float sum = 0.0f;

    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> f32_dist(-10.0f, 10.0f);
    std::uniform_int_distribution<int> i8_dist(-1, 1);

    for (size_t i = 0; i < N; ++i) {
        vec1[i] = f32_dist(gen);
        vec2[i] = static_cast<int8_t>(i8_dist(gen));
        
        sum += vec1[i] * vec2[i];
    }

    return sum;
}

int main(int argc, char** argv) {
    const size_t n = 10000000;
    std::vector<float> vec1;
    std::vector<int8_t> vec2;
    float expected_sum = generate_data(n, vec1, vec2);
    
    std::vector<std::string> failed;

    // 1. Run Tests Once
    for (const auto& kernel : all_match_kernels()) {
        float result = kernel.fn(vec1.data(), vec2.data(), n);
        const float EPSILON = 1; // somewhat generous due to accumulated floating-point errors
        if (std::abs(result - expected_sum) > EPSILON) {
            std::cout << "TEST FAILED: " << kernel.name << std::endl;
            failed.push_back(kernel.name);
        } else {
            std::cout << "TEST PASSED: " << kernel.name << std::endl;
        }
    }

    // 2. Register Benchmarks Once
    for (const auto& kernel : all_match_kernels()) {
        // Skip failures
        if (std::find(failed.begin(), failed.end(), kernel.name) != failed.end()) {
            continue;
        }

        benchmark::RegisterBenchmark(kernel.name.c_str(), [kernel, &vec1, &vec2, n](benchmark::State& state) {
            for (auto _ : state) {
                float res = kernel.fn(vec1.data(), vec2.data(), n);
                benchmark::DoNotOptimize(res);
                benchmark::ClobberMemory(); 
            }
        });
    }

    // 3. Initialize and Run ONCE at the very end
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    
    return 0;
}
