# TensorGraph

A small tensor library in C++20 with a CPU backend (AVX2 + OpenMP), a CUDA
backend, and reverse-mode automatic differentiation on top. The data type and
the device of a tensor are chosen at runtime; the implementation behind each
operation is selected by dynamic dispatch on both.

```cpp
#include "tensor.h"
#include "graph/variable.h"

Tensor x_t = Tensor::from_values({1, 2, 3, 4},
    {.shape = {2, 2}, .device = device_t::GPU, .dtype = dtype_t::Int32});
Tensor y_t = Tensor::from_values({1, 2, 3, 4},
    {.shape = {2, 2}, .device = device_t::GPU, .dtype = dtype_t::Int32});

VariableRef x = Variable::from(x_t);   // graph nodes
VariableRef y = Variable::from(y_t);

VariableRef loss = x->matmul(y)->sum(); // builds the graph
loss->backward();                       // computes gradients

x->grad().to(device_t::CPU);            // d loss / d x, as a CPU tensor
```

## What it does

- **`Tensor`** (`src/tensor.h`): a value type holding a shape, a `dtype_t`
  (`Int32`, `Float32`, `Float64`) and a `device_t` (`CPU`, `GPU`). Constructors
  `from_blob`, `from_values`, `zeroes`, `ones`, `fill`; element-wise `+ - * /`
  (in place and as operators), `sum`, `matmul` and `transpose` for 2-D tensors,
  `copy`, and `to(device)` to move data between host and GPU.
- **Dispatch** (`src/device-tensor.h`, `src/utils.h`): `Tensor` is a pimpl over
  the abstract `DeviceTensor`, implemented once per device. Inside a backend,
  `lift(dtype, f)` turns the runtime `dtype_t` into a template instantiation,
  so each kernel is written once as a template and the `ForEachDType` X-macro
  instantiates it for every supported type.
- **CPU backend** (`src/cpu/`): a `Simd<dtype>` trait wraps the AVX2
  intrinsics for each type; element-wise operations, fills and the reduction
  run 8 or 4 lanes at a time with a scalar tail, and loops above a size
  threshold are parallelised with OpenMP. Matmul and transpose are OpenMP
  loops over the output.
- **GPU backend** (`src/gpu/`): CUDA kernels for fill, element-wise
  operations (one kernel parametrised by a functor), a multi-pass reduction
  with a compile-time reduce width, matmul and transpose. Launch geometry is
  computed per tensor from the device's limits, and device memory is owned by
  a `shared_ptr` with `cudaFree` as its deleter.
- **Autograd** (`src/graph/`): `Variable` nodes hold a value, a gradient, their
  parents and a backward closure that pushes the incoming gradient to the
  parents. `backward()` walks the graph from a root; `cpu()` / `gpu()` move
  an individual node between devices; `clear()` zeroes gradients recursively.
  Supported in the graph: `+ - * /`, `matmul`, `sum`.

## Layout

```
src/            the library
  tensor.h      public Tensor API and the runtime dispatch
  cpu/          AVX2 + OpenMP backend
  gpu/          CUDA backend
  graph/        Variable and the backward pass
tests/          Catch2 tests, every operation on both devices and all dtypes
benchmarks/     Google Benchmark: copy and matmul, serial vs parallel vs GPU
examples/       autograd.cpp, the snippet above as a program
docs/course/    the original assignment texts (in Russian)
```

## Building

Requirements: CMake 3.28+, a C++20 compiler on an AVX2-capable x86-64 machine,
the CUDA toolkit, OpenMP, and for the optional parts
[Catch2 v3](https://github.com/catchorg/Catch2) and
[Google Benchmark](https://github.com/google/benchmark).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build          # tests
./build/benchmarks/benchmark    # benchmarks
./build/examples/autograd       # example
```

Tests, benchmarks and examples can be switched off with
`-DTENSORGRAPH_BUILD_TESTS=OFF`, `-DTENSORGRAPH_BUILD_BENCHMARKS=OFF` and
`-DTENSORGRAPH_BUILD_EXAMPLES=OFF`.

The tests run every operation for `int`, `float` and `double` on a range of
shapes, on the CPU and on the GPU, against straightforward reference
implementations. The autograd test builds a linear layer `x @ w + b`, runs the
backward pass on both devices and checks that activations and gradients agree.

## Origin

Written for the course "Architecture of Machine Learning Platforms" at HSE
University, autumn 2025. The two assignment statements it answers are kept in
`docs/course/`.
