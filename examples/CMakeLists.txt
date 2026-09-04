add_executable(autograd autograd.cpp)

target_link_libraries(autograd PRIVATE graph)
