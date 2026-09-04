#pragma once

#include <immintrin.h>
#include "primitives.h"

template<dtype_t tp>
struct Simd {};

template<>
struct Simd<dtype_t::Float32> {
    using Data = float;
    using simd = __m256;
    static const size_t width = 8;

    static simd load(const Data* data) {
        return _mm256_loadu_ps(data);
    }

    static void store(Data* data, simd reg) {
        _mm256_storeu_ps(data, reg);
    }

    static simd set(Data data) { 
        return _mm256_set1_ps(data);
    }

    static simd add(simd a, simd b) {
        return _mm256_add_ps(a, b);
    }

    static simd subtract(simd a, simd b) {
        return _mm256_sub_ps(a, b);
    }

    static simd multiply(simd a, simd b) {
        return _mm256_mul_ps(a, b);
    }

    static simd divide(simd a, simd b) {
        return _mm256_div_ps(a, b);
    }
};

template<>
struct Simd<dtype_t::Float64> {
    using Data = double;
    using simd = __m256d;
    static const size_t width = 4;

    static simd load(const double* data) {
        return _mm256_loadu_pd(data);
    }

    static void store(double* data, simd reg) {
        _mm256_storeu_pd(data, reg);
    }

    static simd set(Data data) { 
        return _mm256_set1_pd(data);
    }

    static simd add(simd a, simd b) {
        return _mm256_add_pd(a, b);
    }

    static simd subtract(simd a, simd b) {
        return _mm256_sub_pd(a, b);
    }

    static simd multiply(simd a, simd b) {
        return _mm256_mul_pd(a, b);
    }

    static simd divide(simd a, simd b) {
        return _mm256_div_pd(a, b);
    }
};

template<>
struct Simd<dtype_t::Int32> {
    using Data = int32_t;
    using simd = __m256i;
    static const size_t width = 8;

    static simd load(const Data* data) {
        return _mm256_loadu_si256((const __m256i*)data);
    }

    static void store(Data* data, simd reg) {
        _mm256_storeu_si256((simd*)data, reg);
    }

    static simd set(Data elem) { 
        return _mm256_set1_epi32(elem);
    }

    static simd add(simd a, simd b) {
        return _mm256_add_epi32(a, b);
    }

    static simd subtract(simd a, simd b) {
        return _mm256_sub_epi32(a, b);
    }

    static simd multiply(simd a, simd b) {
        return _mm256_mullo_epi32(a, b);
    }

    static simd divide(simd a, simd b) {
        (void)a; (void)b;
        throw std::runtime_error("integer simd division is not supported");
    }
};
