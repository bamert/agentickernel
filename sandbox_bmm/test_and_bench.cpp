#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <random>
#include <algorithm>
#include "registry.hpp"
#include <benchmark/benchmark.h>

// Generates A (Float), B (Packed Binary), and calculates the Golden Reference C.
inline void generate_data(size_t M, size_t K, 
                          std::vector<float>& A, 
                          std::vector<uint32_t>& B, 
                          std::vector<float>& C_expected) {
    
    A.resize(M * K);
    size_t K_ints = K / 32; 
    B.resize(K * K_ints);
    C_expected.assign(M * K, 0.0f);

    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> f32_dist(-5.0f, 5.0f);
    
    // A random uint32_t is exactly 32 random bits!
    std::uniform_int_distribution<uint32_t> u32_dist(0, 0xFFFFFFFF);

    // 1. Populate Matrix A
    for (auto& val : A) {
        val = f32_dist(gen);
    }

    // 2. Populate Matrix B
    for (auto& val : B) {
        val = u32_dist(gen);
    }

    // 3. Calculate C_expected using the textbook triple-loop
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            float sum = 0.0f;
            
            for (size_t p = 0; p < K; ++p) {
                float a_val = A[i * K + p];
                
                // Extract bit at B[p][j]
                uint32_t packed = B[p * K_ints + (j / 32)];
                uint32_t bit = (packed >> (j % 32)) & 1;
                
                float sign = bit ? 1.0f : -1.0f;
                sum += a_val * sign;
            }
            
            C_expected[i * K + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    // Target Dimensions
    const size_t M = 32;
    const size_t K = 3072;

    std::vector<float> A;
    std::vector<uint32_t> B;
    std::vector<float> C_expected;
    
    std::cout << "Generating test matrices (A: " << M << "x" << K << ", B: " << K << "x" << K << ")...\n";
    generate_data(M, K, A, B, C_expected);
    
    // Buffer for the kernels to write into
    std::vector<float> C_target(M * K, 0.0f);
    std::vector<std::string> failed;

    // 1. Run Tests Once
    for (const auto& k : all_match_kernels()) {
        // Clear the target buffer so kernels can't cheat by doing nothing
        std::fill(C_target.begin(), C_target.end(), 0.0f);
        
        // Execute Kernel
        k.fn(A.data(), B.data(), C_target.data(), M, K);
        
        // Validate
        const float EPSILON = 1.0f; // Accumulating 3072 floats introduces drift
        bool pass = true;
        float max_err = 0.0f;
        
        for (size_t i = 0; i < C_expected.size(); ++i) {
            float err = std::abs(C_expected[i] - C_target[i]);
            max_err = std::max(max_err, err);
            if (err > EPSILON) {
                pass = false;
            }
        }

        if (!pass) {
            std::cout << "TEST FAILED: " << k.name << " (Max Error: " << max_err << ")\n";
            failed.push_back(k.name);
        } else {
            std::cout << "TEST PASSED: " << k.name << "\n";
        }
    }

    if (!failed.empty()) {
        std::cerr << "Test failed. Skipping benchmarks " << std::endl;
        exit(1);
    }
    // 2. Register Benchmarks Once
    for (const auto& k : all_match_kernels()) {
        benchmark::RegisterBenchmark(k.name.c_str(), [k, &A, &B, &C_target, M, K](benchmark::State& state) {
            for (auto _ : state) {
                k.fn(A.data(), B.data(), C_target.data(), M, K);
                benchmark::DoNotOptimize(C_target.data());
                benchmark::ClobberMemory(); 
            }
        });
    }

    // 3. Initialize and Run ONCE at the very end
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    
    return 0;
}
