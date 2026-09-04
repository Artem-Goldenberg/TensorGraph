add_library(graph variable.cpp)

target_include_directories(graph PRIVATE ..)
target_link_libraries(graph PUBLIC tensor)
