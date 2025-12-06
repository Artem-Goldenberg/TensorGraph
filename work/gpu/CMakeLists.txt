enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(gpu SHARED tensor.cu)

target_include_directories(gpu PRIVATE ..)

set_target_properties(gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
