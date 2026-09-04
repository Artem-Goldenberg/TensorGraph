cmake_minimum_required(VERSION 3.28)
project(TensorGraph VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TENSORGRAPH_BUILD_TESTS "Build the Catch2 test suite" ON)
option(TENSORGRAPH_BUILD_BENCHMARKS "Build the Google Benchmark suite" ON)
option(TENSORGRAPH_BUILD_EXAMPLES "Build the example programs" ON)

add_compile_options(
    -mavx2
    -Wall -Wextra
    $<$<CONFIG:RELEASE>:-O3>
    $<$<CONFIG:DEBUG>:-O0>
    $<$<CONFIG:DEBUG>:-ggdb3>
    $<$<CONFIG:DEBUG>:-g>
)

add_subdirectory(src)

if(TENSORGRAPH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TENSORGRAPH_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(TENSORGRAPH_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
