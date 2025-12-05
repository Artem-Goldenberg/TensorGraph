#pragma once

#include <cassert>

#include "primitives.h"

#define ForEachDType(X) \
    X(dtype_t::Float32, float) \
    X(dtype_t::Float64, double) \
    X(dtype_t::Int32,   int)

template <typename Fn>
constexpr void lift(dtype_t type, Fn&& func) {
    #define Case(tp, T) \
        case tp: \
            func.template operator()<tp>(); \
            break;

    switch (type) {
        ForEachDType(Case)
    default:
        assert(false);
    }
}

template <dtype_t tp>
struct Info {};

#define InfoDef(tp, T) \
    template<> \
    struct Info<tp> { \
        using Data = T; \
    };

ForEachDType(InfoDef)

template <typename Data>
constexpr dtype_t unlift;

#define UnliftDef(tp, T) \
    template<> constexpr dtype_t unlift<T> = tp;

ForEachDType(UnliftDef)

constexpr size_t size_of(dtype_t type) {
    size_t result = 0;
    lift(type, [&result]<dtype_t tp>() {
        result = sizeof(typename Info<tp>::Data);
    });
    return result;
}

constexpr void check(bool cond, std::string_view message) {
    if (!cond) {
        throw std::runtime_error(std::string(message));
    }    
}
