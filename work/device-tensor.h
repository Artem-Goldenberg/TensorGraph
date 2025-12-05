#pragma once

#include <memory>
#include <stdexcept>

#include "primitives.h"

class DeviceTensor;

using TensorRef = std::shared_ptr<DeviceTensor>;

class DeviceTensor {
protected:
    TensorParams params;
    std::shared_ptr<void> data = nullptr;

    DeviceTensor(const TensorParams& params): params(params) {}

public:
    const TensorParams& get_params() const {
        return params;
    };

    template <typename Data>
    const Data* get_data() const {
        return static_cast<const Data*>(data);
    };

    template <typename Data>
    Data* get_mutable_data() {
        return static_cast<Data*>(data);
    };

    virtual TensorRef copy() const = 0;

    virtual void add(const DeviceTensor& other) = 0;
    virtual void subtract(const DeviceTensor& other) = 0;
    virtual void multiply(const DeviceTensor& other) = 0;
    virtual void divide(const DeviceTensor& other) = 0;
};
