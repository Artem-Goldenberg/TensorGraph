#include <benchmark/benchmark.h>

#include "benchmarks/utils.h"
#include "../utils.h"

#include <iostream>

#include "primitives.h"
#include "tensor.h"

using std::vector;

constexpr size_t seed = 42;

static const vector<vector<size_t>> problems = {
    {2, 2, 2, 2},
    {1024, 1024, 3},
    {1024, 1024}
};

inline void supply_data(benchmark::internal::Benchmark* b) {
    for (size_t i = 0; i < problems.size(); ++i) {
        b->Args({(int64_t)i});
    }
}

template <typename Data>
static void copy(benchmark::State& state) {
    int64_t i = state.range(0);
    Shape shape = Shape(problems[i]);

    auto N = state.range(0);
    auto [a_data, b_data] = make_test_data<Data>(seed, N);

    Tensor a = Tensor::from_blob(a_data, {.shape = shape, .dtype = unlift<Data>});
    Tensor b = Tensor::from_blob(b_data, {.shape = shape, .dtype = unlift<Data>});

    std::vector<Data> result(N, 0);

    for (auto _ : state) {
        Tensor throwaway = a.copy();
        benchmark::DoNotOptimize(throwaway);
        benchmark::ClobberMemory();
    }
}

// copying
// subtraction, multiplication, division
// matrix multiplication

#define DoBenchmark(tp, T) \
    BENCHMARK_TEMPLATE1(copy, T) \
        ->Apply(supply_data) \
        ->Unit(benchmark::kMicrosecond);

ForEachDType(DoBenchmark)

// BENCHMARK_TEMPLATE1(addition, float)
//     ->Apply(problems)
//     ->Unit(benchmark::kMicrosecond);
// BENCHMARK_TEMPLATE1(addition, double)
//     ->Apply(problems)
//     ->Unit(benchmark::kMicrosecond);
