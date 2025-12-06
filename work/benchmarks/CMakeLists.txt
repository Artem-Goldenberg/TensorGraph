find_package(benchmark REQUIRED)

add_executable(benchmark cpu_benchmark.cpp)

target_include_directories(benchmark PRIVATE ..)

target_link_libraries(benchmark benchmark::benchmark benchmark::benchmark_main tensor cpu gpu)

