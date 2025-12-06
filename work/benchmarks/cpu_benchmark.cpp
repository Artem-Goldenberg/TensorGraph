#include <benchmark/benchmark.h>

#include "benchmarks/utils.h"
#include "../utils.h"

#include <iostream>

#include "primitives.h"
#include "tensor.h"

using std::vector;

constexpr size_t seed = 42;

template <size_t n>
inline void range(benchmark::internal::Benchmark* b) {
    for (size_t i = 0; i < n; ++i) {
        b->Args({(int64_t)i});
    }
}

static const vector<size_t> problems[] = {
    {2, 2, 2, 2},
    {1024, 1024, 3},
    {1024, 1024}
};

constexpr size_t n_problems = sizeof(problems) / sizeof(problems[0]);

template <typename Data>
static void copy(benchmark::State& state) {
    int64_t i = state.range(0);
    Shape shape = Shape(problems[i]);

    auto [a_data, b_data] = make_test_data<Data>(seed, shape.numel());

    Tensor a = Tensor::from_blob(a_data, {.shape = shape, .dtype = unlift<Data>});
    // Tensor b = Tensor::from_blob(b_data, {.shape = shape, .dtype = unlift<Data>});

    // std::vector<Data> result(N, 0);

    for (auto _ : state) {
        Tensor throwaway = a.copy();
        benchmark::DoNotOptimize(throwaway);
        benchmark::ClobberMemory();
    }
}

static const vector<size_t> matmul_problems[] = {
    {1024, 1024, 3},
    {1024, 1024, 1024},
    {1024, 1024, 2048}
};

constexpr size_t n_matmul_problems = sizeof(matmul_problems) / sizeof(matmul_problems[0]);

template <typename Data>
static void matmul(benchmark::State& state, device_t device) { 
    int64_t i = state.range(0);

    vector<size_t> dims_pack = matmul_problems[i];

    Shape a_shape = vector<size_t>(dims_pack.begin(), dims_pack.begin() + 2);
    Shape b_shape = vector<size_t>(dims_pack.begin() + 1, dims_pack.end());

    assert(a_shape.ndims() == 2);
    assert(b_shape.ndims() == 2);
    assert(a_shape[1] == b_shape[0]);

    Data* a_data = new Data[a_shape.numel()];
    Data* b_data = new Data[b_shape.numel()];

    std::mt19937 rng(seed);

    fill_random_data<Data>(rng, a_data, a_shape.numel());
    fill_random_data<Data>(rng, b_data, b_shape.numel());

    Tensor a = Tensor::from_blob<Data>(
        a_data, {.shape = a_shape, .device = device, .dtype = unlift<Data>}
    );
    Tensor b = Tensor::from_blob<Data>(
        b_data, {.shape = b_shape, .device = device, .dtype = unlift<Data>}
    );

    delete[] a_data;
    delete[] b_data;

    for (auto _ : state) {
        Tensor c = a.matmul(b);
        benchmark::DoNotOptimize(c);
        benchmark::ClobberMemory();
    }
}

template <typename Data>
static void cpu_matmul(benchmark::State& state) {
    matmul<Data>(state, device_t::CPU);
}

template <typename Data>
static void gpu_matmul(benchmark::State& state) {
    matmul<Data>(state, device_t::GPU);
}

// copying
// subtraction, multiplication, division
// matrix multiplication

#define CopyBenchmark(tp, T) \
    BENCHMARK_TEMPLATE1(copy, T) \
        ->Apply(range<n_problems>) \
        ->Unit(benchmark::kMicrosecond);

ForEachDType(CopyBenchmark)

#define CPUMatmulBenchmark(tp, T) \
    BENCHMARK_TEMPLATE1(cpu_matmul, T) \
        ->Apply(range<n_matmul_problems>) \
        ->Unit(benchmark::kMicrosecond);

#define GPUMatmulBenchmark(tp, T) \
    BENCHMARK_TEMPLATE1(gpu_matmul, T) \
        ->Apply(range<n_matmul_problems>) \
        ->Unit(benchmark::kMicrosecond);

ForEachDType(CPUMatmulBenchmark)
ForEachDType(GPUMatmulBenchmark)

// BENCHMARK_TEMPLATE1(addition, float)
//     ->Apply(problems)
//     ->Unit(benchmark::kMicrosecond);
// BENCHMARK_TEMPLATE1(addition, double)
//     ->Apply(problems)
//     ->Unit(benchmark::kMicrosecond);
