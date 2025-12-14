find_package(OpenMP REQUIRED)

add_library(cpu tensor.cpp)

target_link_libraries(cpu PRIVATE OpenMP::OpenMP_CXX)

target_include_directories(cpu PRIVATE ..)

