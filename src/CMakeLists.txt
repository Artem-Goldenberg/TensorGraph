# The library is split into the device backends (cpu, gpu), the device-agnostic
# Tensor facade that dispatches to them (tensor), and the autograd graph built
# on top of it (graph). Link against `graph` to get everything.
add_subdirectory(cpu)
add_subdirectory(gpu)

add_library(tensor tensor.cpp)
target_include_directories(tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tensor PUBLIC cpu gpu)

add_subdirectory(graph)
