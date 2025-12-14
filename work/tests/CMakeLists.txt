find_package(Catch2 REQUIRED)
include(Catch)

find_package(OpenMP REQUIRED)

add_executable(tests tests.cpp graph_test.cpp)

target_include_directories(tests PRIVATE ..)

target_link_libraries(tests PRIVATE tensor cpu gpu graph Catch2::Catch2WithMain OpenMP::OpenMP_CXX)

catch_discover_tests(tests)
