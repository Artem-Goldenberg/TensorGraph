#pragma once

#include <cmath>
#include <vector>
#include <stdexcept>
#include <catch2/catch_all.hpp>

#include "benchmarks/utils.h"

constexpr size_t seed = 42;

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
                printf("Some at %ld\n", i);
            REQUIRE(actual[i] == expected[i]);
        }
    }
}

template <typename Data>
using Matrix = vector<vector<Data>>;

template <typename Data>
Matrix<Data> fullproof_matmul(const Matrix<Data>& a, const Matrix<Data>& b) {
    // Check if either matrix is empty
    if (a.empty() || b.empty()) 
        throw std::invalid_argument("Matrices must not be empty.");

    // Check if all rows in a have same size
    size_t a_cols = a[0].size();
    for (const auto& row : a) 
        if (row.size() != a_cols)
            throw std::invalid_argument("All rows in matrix A must have same size.");

    // Check if all rows in b have same size
    size_t b_cols = b[0].size();
    size_t b_rows = b.size();
    for (const auto& row : b) 
        if (row.size() != b_cols)
            throw std::invalid_argument("All rows in matrix B must have same size.");

    // Check multiplication compatibility
    if (a_cols != b_rows)
        throw std::invalid_argument("Number of columns of A must equal number of rows of B.");

    // Create result matrix with zeros
    Matrix<Data> result(a.size(), vector<Data>(b_cols, Data{}));

    // Perform multiplication
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b_cols; ++j) {
            Data sum = Data{};
            for (size_t k = 0; k < a_cols; ++k) {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }

    return result;
}
