find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(benchmark benchmark.cpp)

target_link_libraries(benchmark PRIVATE )

target_include_directories(benchmark PRIVATE ..)

target_link_libraries(benchmark 
    benchmark::benchmark 
    benchmark::benchmark_main 
    OpenMP::OpenMP_CXX
    tensor cpu gpu
)
