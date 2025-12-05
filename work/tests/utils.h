#pragma once

#include <cmath>
#include <catch2/catch_all.hpp>

#include "benchmarks/utils.h"

constexpr size_t seed = 42;

template <typename TT, size_t NN> struct TestCase {
  using T = TT;
  static constexpr size_t N = NN;
};

template <size_t NN> using Int32Case = TestCase<int32_t, NN>;
template <size_t NN> using Float32Case = TestCase<float, NN>;
template <size_t NN> using Float64Case = TestCase<double, NN>;

#define TEST_VARIANTS                                                          \
  ((size_t N), N), (Int32Case, Float32Case, Float64Case),                      \
      ((8), (32), (512), (4096), (1024 * 1024))


template <typename T>
bool nearly_equal_impl(
    T a, T b, T max_rel_diff = T(0.01), T max_diff = T(1e-7)
) {
    if (std::isinf(a) || std::isinf(b))
        return a == b;   // +inf == +inf, -inf == -inf

    // Handle NaNs explicitly
    const bool a_nan = std::isnan(a);
    const bool b_nan = std::isnan(b);
    if (a_nan && b_nan)
        return true;     // NaN == NaN in this comparison scheme
    if (a_nan || b_nan)
        return false;    // one is NaN, the other isn't

    // Regular comparison
    T diff = std::fabs(a - b);
    if (diff <= max_diff)
        return true;

    a = std::fabs(a);
    b = std::fabs(b);
    T largest = (a > b) ? a : b;

    if (diff <= largest * max_rel_diff)
        return true;

    return false;
}

template <typename T>
void equals(
    const vector<T>& expected,
    const T* actual,
    T max_rel_diff = 0.01,
    T max_diff = 0.0000001
) {
    for (size_t i = 0; i < expected.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            INFO(i);
            INFO(actual[i]);
            INFO(expected[i]);
            // if (!nearly_equal_impl(actual[i], expected[i], max_rel_diff, max_diff)) {
            //     printf("i = %d\n", i);
            // }
            REQUIRE(nearly_equal_impl(actual[i], expected[i], max_rel_diff, max_diff));
        } else {
            if (actual[i] != expected[i])
                printf("Some at %d\n", i);
            REQUIRE(actual[i] == expected[i]);
        }
    }
}
