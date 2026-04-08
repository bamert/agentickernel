cmake_minimum_required(VERSION 3.14)
project(KernelEvaluator CXX)

# Enforce C++17 (required by modern Google Test/Benchmark)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the testing and benchmarking libraries 
# (Assumes they are installed system-wide via apt, brew, or vcpkg)
find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)

add_executable(test test.cpp)

target_link_libraries(test
    PRIVATE 
    GTest::gtest_main
    GTest::gtest
    benchmark::benchmark
    pthread
)
