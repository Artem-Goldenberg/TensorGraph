#include <benchmark/benchmark.h>

#include "omp.h"
#include "bench_utils.h"
#include "utils.h"

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
    {16, 16},
    {2048, 128},
    {128, 2048},
    {2048, 1024, 3},
    {1024, 16, 1024},
    {16, 2048, 2048},
    {101, 107, 103, 105}
};

static const vector<size_t> matmul_problems[] = {
    {16, 32, 16},
    {1024, 1024, 3},
    {1024, 1024, 1024},
    {1024, 1024, 2048},
    {1007, 1003, 2047}
};

constexpr size_t n_problems = sizeof(problems) / sizeof(problems[0]);

template <typename Data>
static void copy(benchmark::State& state, device_t device = device_t::CPU) {
    int64_t i = state.range(0);
    Shape shape = Shape(problems[i]);

    auto [a_data, b_data] = make_test_data<Data>(seed, shape.numel());

    Tensor a = Tensor::from_blob(
        a_data, {.shape = shape, .device = device, .dtype = unlift<Data>}
    );

    for (auto _ : state) {
        Tensor throwaway = a.copy();
        benchmark::DoNotOptimize(throwaway);
        benchmark::ClobberMemory();
    }
}

template <typename Data>
static void serial_copy(benchmark::State& state) {
    omp_set_dynamic(0);
    omp_set_num_threads(1);
    copy<Data>(state);
}

template <typename Data>
static void parallel_copy(benchmark::State& state) {
    omp_set_dynamic(0);
    omp_set_num_threads(omp_get_num_procs());

    // Warm up the threads
    #pragma omp parallel
    {}

    copy<Data>(state);
}

template <typename Data>
static void gpu_copy(benchmark::State& state) {
    copy<Data>(state, device_t::GPU);
}

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
static void cpu_serial_matmul(benchmark::State& state) {
    omp_set_dynamic(0);
    omp_set_num_threads(1);
    matmul<Data>(state, device_t::CPU);
}

template <typename Data>
static void cpu_parallel_matmul(benchmark::State& state) {
    omp_set_dynamic(0);
    omp_set_num_threads(omp_get_num_procs());

    // Warm up the threads
    #pragma omp parallel
    {}

    matmul<Data>(state, device_t::CPU);
}

template <typename Data>
static void gpu_matmul(benchmark::State& state) {
    matmul<Data>(state, device_t::GPU);
}

// copying
// subtraction, multiplication, division
// matrix multiplication

#define SerialCopyBenchmark(tp, T) \
    BENCHMARK_TEMPLATE1(serial_copy, T) \
        ->Apply(range<n_problems>) \
        ->Unit(benchmark::kMicrosecond);

#define ParallelCopyBenchmark(tp, T) \
    BENCHMARK_TEMPLATE1(parallel_copy, T) \
        ->Apply(range<n_problems>) \
        ->Unit(benchmark::kMicrosecond);

#define GPUCopyBenchmark(tp, T) \
    BENCHMARK_TEMPLATE1(gpu_copy, T) \
        ->Apply(range<n_problems>) \
        ->Unit(benchmark::kMicrosecond);

ForEachDType(SerialCopyBenchmark)
ForEachDType(ParallelCopyBenchmark)
ForEachDType(GPUCopyBenchmark)

#define CPUSerialMatmulBenchmark(tp, T) \
    BENCHMARK_TEMPLATE1(cpu_serial_matmul, T) \
        ->Apply(range<n_matmul_problems>) \
        ->Unit(benchmark::kMicrosecond);

#define CPUParallelMatmulBenchmark(tp, T) \
    BENCHMARK_TEMPLATE1(cpu_parallel_matmul, T) \
        ->Apply(range<n_matmul_problems>) \
        ->Unit(benchmark::kMicrosecond);

#define GPUMatmulBenchmark(tp, T) \
    BENCHMARK_TEMPLATE1(gpu_matmul, T) \
        ->Apply(range<n_matmul_problems>) \
        ->Unit(benchmark::kMicrosecond);

ForEachDType(CPUSerialMatmulBenchmark)
ForEachDType(CPUParallelMatmulBenchmark)
ForEachDType(GPUMatmulBenchmark)

// BENCHMARK_TEMPLATE1(addition, float)
//     ->Apply(problems)
//     ->Unit(benchmark::kMicrosecond);
// BENCHMARK_TEMPLATE1(addition, double)
//     ->Apply(problems)
//     ->Unit(benchmark::kMicrosecond);
