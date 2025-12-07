#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>

enum class device_t {
    CPU, GPU
};

enum class dtype_t {
    Int32, Float32, Float64
};

struct Shape {
    Shape() {}
    Shape(std::initializer_list<size_t> dims): dims(dims) {}
    Shape(std::vector<size_t> dims): dims(dims) {}

    constexpr const std::vector<size_t>& get_dims() const { 
        return dims;
    }

    constexpr size_t ndims() const {
        return dims.size();
    }

    constexpr size_t operator[] (int index) const {
        if (index >= 0)
            return dims[index];
        return dims[dims.size() + index];
    }

    constexpr bool operator == (const Shape& other) const {
        return std::equal(dims.begin(), dims.end(), other.dims.begin());
    }

    constexpr bool operator != (const Shape& other) const {
        return !(*this == other);
   }

    constexpr size_t numel() const {
        size_t result = 1;
        for (size_t dim : dims)
            result *= dim;
        return result;
    }

private:
    std::vector<size_t> dims;
};

struct TensorParams {
    Shape shape;
    device_t device = device_t::CPU;
    dtype_t dtype = dtype_t::Float32;

    TensorParams copy() const {
        return TensorParams(*this);
    }

    TensorParams& with_shape(const Shape& shape) {
        this->shape = shape;
        return *this;
    }

    TensorParams& with_device(device_t device) {
        this->device = device;
        return *this;
    }

    TensorParams& with_dtype(dtype_t dtype) {
        this->dtype = dtype;
        return *this;
    }
};
