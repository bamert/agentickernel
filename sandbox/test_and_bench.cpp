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
int main(int argc, char** argv){
    const size_t BENCH_N = 100000; 
    std::vector<float> bench_vec1;
    std::vector<int8_t> bench_vec2;
    
    // Pass nullptr (default) since we don't need the sum for benchmarking
    generate_data(BENCH_N, bench_vec1, bench_vec2);
    
    // Run baseline
    for (const auto& kernel : all_match_kernels()) {
        const size_t n = 10000;
        std::vector<float> vec1;
        std::vector<int8_t> vec2;
        
        float expected_sum = generate_data(n, vec1, vec2);
        float result = kernel.fn(vec1.data(), vec2.data(), n);

        const float EPSILON = 1e-5f;
        if (std::abs(result - expected_sum) > EPSILON) {
            std::cerr << "TEST FAILED for kernel: " << kernel.name << std::endl;
            return 1;
        } else {
            std::cout << "TEST PASSED for kernel: " << kernel.name << std::endl;
        }
        for (const auto& kernel : all_match_kernels()) {
            benchmark::RegisterBenchmark(kernel.name.c_str(), [kernel, &vec1, &vec2, n](benchmark::State& state) {
            for (auto _ : state) {
                float res = kernel.fn(vec1.data(), vec2.data(), n);
                benchmark::DoNotOptimize(res);
                benchmark::ClobberMemory(); 
            }
        });
    }
        benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    }
    return 0;
}
