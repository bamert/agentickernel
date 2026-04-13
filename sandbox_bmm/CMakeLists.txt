cmake_minimum_required(VERSION 3.14)
project(KernelEvaluator CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mcpu=native -DNDEBUG")

find_package(benchmark REQUIRED)

add_executable(test_and_bench test_and_bench.cpp)

target_link_libraries(test_and_bench
    PRIVATE 
    benchmark::benchmark
    pthread
)
