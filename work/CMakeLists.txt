

include(FetchContent)

FetchContent_Declare(
    libtorch
    # URL https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.9.1%2Bcu128.zip
    # URL https://download.pytorch.org/libtorch/cu128/libtorch-cxx20-abi-shared-with-deps-2.9.1+cu128.zip
    # URL https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.8.0%2Bcpu.zip
    # URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.9.1.zip
    URL https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.8.0%2Bcpu.zip
)

# find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
# include(Python)
# message(STATUS "Python executable: ${Python3_EXECUTABLE}")
# message(STATUS "Python include dirs: ${Python3_INCLUDE_DIRS}")

# FetchContent_Declare(
#     python
#     URL https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz
#     URL_HASH SHA256=<hash>
# )

# FetchContent_GetProperties(python)
# if(NOT python_POPULATED)
#     FetchContent_Populate(python)

#     # python_SOURCE_DIR now has the source
#     # You could build it manually using add_custom_command / add_custom_target
#     add_custom_target(python_build ALL
#         COMMAND ./configure --prefix=${CMAKE_BINARY_DIR}/python
#         COMMAND make -j
#         COMMAND make install
#         WORKING_DIRECTORY ${python_SOURCE_DIR}
#     )
# endif()

FetchContent_MakeAvailable(libtorch)

set(Torch_DIR "${libtorch_SOURCE_DIR}/share/cmake/Torch")
find_package(Torch REQUIRED)

add_subdirectory(cpu)
add_subdirectory(gpu)
add_subdirectory(graph)
add_subdirectory(benchmarks)
add_subdirectory(tests)

add_executable(main main.cpp tensor.cpp)

add_library(tensor tensor.cpp)
# target_link_libraries(tensor PRIVATE cpu)
target_include_directories(tensor PRIVATE .)

target_link_libraries(main "${TORCH_LIBRARIES}" cpu gpu graph)
target_compile_definitions(main PRIVATE TORCH_API_INCLUDE_EXTENSION_H)
target_include_directories(main PRIVATE "${TORCH_INCLUDE_DIRS}")

target_include_directories(main PRIVATE .)
