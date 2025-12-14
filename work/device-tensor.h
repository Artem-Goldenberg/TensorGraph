#pragma once

#include <memory>
#include <stdexcept>

#include "primitives.h"
#include "utils.h"

class DeviceTensor;

using TensorRef = std::shared_ptr<DeviceTensor>;

class DeviceTensor {
protected:
    TensorParams params;
    std::shared_ptr<void> data = nullptr;

    DeviceTensor(const TensorParams& params): params(params) {}

public:
    constexpr const TensorParams& get_params() const {
        return params;
    };

    template <typename Data>
    const Data* get_data() const {
        return static_cast<const Data*>(data.get());
    };

    template <typename Data>
    Data* get_mutable_data() {
        return static_cast<Data*>(data.get());
    };

    virtual TensorRef copy() const = 0;

    virtual void add(const DeviceTensor& other) = 0;
    virtual void subtract(const DeviceTensor& other) = 0;
    virtual void multiply(const DeviceTensor& other) = 0;
    virtual void divide(const DeviceTensor& other) = 0;

    virtual TensorRef sum() const = 0;

    virtual TensorRef matmul(const DeviceTensor& other) const = 0;
    virtual TensorRef transpose() const = 0;

    virtual void clear() = 0;

    virtual ~DeviceTensor() = default;
};

constexpr void validate(const Shape& shape) {
    check(shape.ndims() > 0, "Tensor cannot have an empty shape");

    for (size_t i = 0; i < shape.ndims(); ++i)
        check(shape[i] > 0, "Cannot have dimensions with 0 size");
}

constexpr void check_compatible(const DeviceTensor& self, const DeviceTensor& other) {
    const TensorParams& params = self.get_params();
    const TensorParams& other_params = other.get_params();

    check(params.device == other_params.device,
        "Tried to do an operation on two tensors from different devices");
    
    check(params.dtype == other_params.dtype,
        "Cannot do an operation on tensors of different types");
    
    // Also validate the shape while we are here
    validate(params.shape);
    validate(other_params.shape);
}

constexpr void check_bulk_compatible(const DeviceTensor& self, const DeviceTensor& other) {
    check_compatible(self, other);

    check(self.get_params().shape == other.get_params().shape, 
        "Cannot do a bulk operation on tensors of different shape");
}

constexpr void check_matmul_compatible(const DeviceTensor& self, const DeviceTensor& other) { 
    check_compatible(self, other);

    const Shape& shape = self.get_params().shape;
    const Shape& other_shape = other.get_params().shape;

    check(shape.ndims() == 2 && other_shape.ndims() == 2,
        "Matmul operations on non-2D tensors are not supported");
    
    check(shape[1] == other_shape[0],
        "Mismatch on the common dimension in matmul");
}

constexpr void check_transposable(const DeviceTensor& self) { 
    const Shape& shape = self.get_params().shape;

    validate(shape);

    check(shape.ndims() == 2, "Transpose for non-2D tensors is not supported");
}
