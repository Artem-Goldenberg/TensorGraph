add_subdirectory(cpu)
add_subdirectory(gpu)
add_subdirectory(benchmarks)
add_subdirectory(tests)

add_executable(tensor_build main.cpp tensor.cpp)

add_library(tensor tensor.cpp)
# target_link_libraries(tensor PRIVATE cpu)
target_include_directories(tensor PRIVATE .)

target_link_libraries(tensor_build PRIVATE cpu gpu)

target_include_directories(tensor_build PRIVATE .)
