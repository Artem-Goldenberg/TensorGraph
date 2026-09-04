find_package(Catch2 3 REQUIRED)
include(Catch)

find_package(OpenMP REQUIRED)

add_executable(tests tests.cpp graph_test.cpp)

# The project root is on the include path so tests can reach benchmarks/bench_utils.h.
target_include_directories(tests PRIVATE ${PROJECT_SOURCE_DIR})

target_link_libraries(tests PRIVATE graph Catch2::Catch2WithMain OpenMP::OpenMP_CXX)

catch_discover_tests(tests)
