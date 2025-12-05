cmake_minimum_required(VERSION 3.28)
project(homework VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

add_compile_options(
    # -g
    -O3
    -mavx2
    -Wall -Wextra 
    # -fsanitize=undefined 
    # -fno-sanitize-recover=all
    # -fno-sanitize=signed-integer-overflow
    # $<$<CONFIG:RELEASE>:-O3>
    # $<$<CONFIG:DEBUG>:-O0>
    # $<$<CONFIG:DEBUG>:-ggdb3>
)

add_link_options(
    # -fsanitize=undefined
    # -fno-sanitize=signed-integer-overflow
)

add_subdirectory(work)
add_subdirectory(testing)

# add_subdirectory(tests/part1)
# add_subdirectory(benchmarks/part1)

# add_subdirectory(tests/part2)
# add_subdirectory(benchmarks/part2)

# add_subdirectory(tests/part3)
# add_subdirectory(benchmarks/part3)
