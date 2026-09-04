#pragma once

#include <algorithm>
#include <random>
#include <tuple>
#include <vector>
#include <memory>
#include <cassert>

using std::vector;
using std::unique_ptr;

template <typename Data, typename Rng>
void fill_random_data(Rng& rng, Data* data, size_t n) {
    if constexpr (std::is_floating_point_v<Data>) {
        std::uniform_real_distribution<Data> dist(0, 1.);
        for (size_t i = 0; i < n; ++i)
            data[i] = dist(rng);
    } else {
        std::uniform_int_distribution<Data> dist(1, n);
        for (size_t i = 0; i < n; ++i)
            data[i] = dist(rng);
    }
}

template <typename Data>
std::tuple<Data*, Data*> make_test_data(size_t seed, size_t num_elements) {
    std::mt19937 rng(seed);

    auto x = new Data[num_elements];
    auto y = new Data[num_elements];

    assert(x);
    assert(y);

    fill_random_data<Data>(rng, x, num_elements);
    fill_random_data<Data>(rng, y, num_elements);

    return { x, y };
}
