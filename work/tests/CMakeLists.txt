find_package(Catch2 REQUIRED)
include(Catch)

add_executable(tests cpu_tests.cpp)

target_include_directories(tests PRIVATE ..)

target_link_libraries(tests PRIVATE tensor cpu Catch2::Catch2WithMain)

catch_discover_tests(tests)
