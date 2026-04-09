cmake_minimum_required(VERSION 3.14)
project(KernelEvaluator CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)

add_executable(test_and_bench test_and_bench.cpp)

target_link_libraries(test_and_bench
    PRIVATE 
    GTest::gtest_main
    GTest::gtest
    benchmark::benchmark
    pthread
)
