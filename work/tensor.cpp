#include <cstring>
#include "tensor.h"

#include "cpu/tensor.h"
#include "utils.h"

using std::vector;
using std::tuple;
using std::shared_ptr;
using std::unique_ptr;
using std::runtime_error;

namespace {
    template <typename Data>
    shared_ptr<DeviceTensor> make_tensor(const Data* data, TensorParams params) {
        switch (params.device)
        {
        case device_t::CPU:
            return std::make_shared<cpu::Tensor>(params, data);
        case device_t::GPU:
        default:
            throw std::runtime_error("Unsupported device");
        }
    }
}

template<typename Data>
Tensor Tensor::from_blob(const Data* data, TensorParams params) {
    return Tensor(make_tensor(data, params));
}

Tensor Tensor::zeroes(TensorParams params) {
    return Tensor::fill(0, params);
}

Tensor Tensor::ones(TensorParams params) {
    return Tensor::fill(1, params);
}

template <typename Data>
Tensor Tensor::fill(Data some, TensorParams params) {
    if (unlift<Data> != params.dtype)
        throw runtime_error("Bad dtype to fill the tensor with");

    size_t n = params.shape.numel();
    shared_ptr<DeviceTensor> impl;

    lift(params.dtype, [&]<dtype_t tp>() {
        unique_ptr<Data> temp = std::make_unique<Data>(n);
        std::memset(temp.get(), some, n);
        impl = make_tensor(temp.get(), params);
    });

    return Tensor(impl);
}

template<typename Data>
const Data* Tensor::get_data() const {
    return pImpl->get_data<Data>();
}

template<typename Data>
Data* Tensor::get_mutable_data() {
    return pImpl->get_mutable_data<Data>();
}

#define InstantiateTemplates(tp, Data) \
    template Tensor Tensor::from_blob<Data>(const Data*, TensorParams); \
    template Tensor Tensor::fill<Data>(Data, TensorParams); \
    template const Data* Tensor::get_data<Data>() const; \
    template Data* Tensor::get_mutable_data<Data>();

ForEachDType(InstantiateTemplates)

const TensorParams& Tensor::get_params() const {
    return pImpl->get_params();
}

Tensor Tensor::copy() const {
    return Tensor(pImpl->copy());
}

void Tensor::add(const Tensor& other) { pImpl->add(*other.pImpl); }
void Tensor::subtract(const Tensor& other) { pImpl->subtract(*other.pImpl); }
void Tensor::multiply(const Tensor& other) { pImpl->multiply(*other.pImpl); }
void Tensor::divide(const Tensor& other) { pImpl->divide(*other.pImpl); }

Tensor& Tensor::operator+=(const Tensor& other) {
    this->add(other);
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    this->subtract(other);
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    this->multiply(other);
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    this->divide(other);
    return *this;
}
