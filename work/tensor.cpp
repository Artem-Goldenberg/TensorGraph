#include <cstring>
#include <iostream>
#include <functional>

#include "tensor.h"

#include "cpu/tensor.h"
#include "gpu/tensor.h"
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
            return std::make_shared<gpu::Tensor>(params, data);
        default:
            throw std::runtime_error("Unsupported device");
        }
    }
}

template<typename Data>
Tensor Tensor::from_blob(const Data* data, TensorParams params) {
    return Tensor(make_tensor<Data>(data, params));
}

template<typename Data>
Tensor Tensor::from_values(std::initializer_list<Data> values, TensorParams params) {
    Data* data = new Data[values.size()];
    std::copy(values.begin(), values.end(), data);

    Tensor result = Tensor(make_tensor<Data>(data, params));

    delete[] data;
    return result;
}

Tensor Tensor::zeroes(TensorParams params) {
    return lift(params.dtype, [&]<dtype_t tp>() -> Tensor {
        return Tensor::fill(Info<tp>::zero, params);
    });
}

Tensor Tensor::ones(TensorParams params) {
    Tensor* result_ptr = nullptr;
    lift(params.dtype, [&]<dtype_t tp>() {
        Tensor result = Tensor::fill(Info<tp>::one, params);
        result_ptr = &result;
    });
    return *result_ptr;
}

template <typename Data>
Tensor Tensor::fill(Data some, TensorParams params) {
    if (unlift<Data> != params.dtype)
        throw runtime_error("Bad dtype to fill the tensor with");

    size_t n = params.shape.numel();

    Data* data = new Data[n];
    std::memset(data, some, n);

    TensorRef impl = make_tensor(data, params);

    delete[] data;

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
    template Tensor Tensor::from_values<Data>(std::initializer_list<Data>, TensorParams); \
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

Tensor Tensor::to(device_t device) const {
    if (this->get_params().device == device) {
        // std::cout << "Tensor device " << device << " haven't changed" << std::endl;
        return *this;
    }

    shared_ptr<DeviceTensor> new_impl;
    TensorParams old_params = get_params();

    lift(old_params.dtype, [&]<dtype_t tp>() {
        using Data = Info<tp>::Data;
    
        if (device == device_t::CPU) {
            // Gone from GPU to CPU, need to flush the memory from GPU
            new_impl = make_tensor<Data>(nullptr, old_params.copy().with_device(device));
            shared_ptr<gpu::Tensor> impl = std::dynamic_pointer_cast<gpu::Tensor>(pImpl);
            assert(impl);
            impl->flush<Data>(new_impl->get_mutable_data<Data>());
        } 
        else {
            // Gone from CPU to GPU, just create a gpu tensor
            new_impl = make_tensor<Data>(
                pImpl->get_data<Data>(), old_params.copy().with_device(device)
            );
        }
    });

    return Tensor(new_impl);
}

Tensor Tensor::sum() const {
    return Tensor(pImpl->sum());
}

Tensor Tensor::matmul(const Tensor& other) const {
    return Tensor(pImpl->matmul(*other.pImpl));
}

Tensor Tensor::transpose() const {
    return Tensor(pImpl->transpose());
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

Tensor Tensor::operator + (const Tensor& other) const {
    Tensor result = copy();
    result += other;
    return result;
}

Tensor Tensor::operator - (const Tensor& other) const {
    Tensor result = copy();
    result -= other;
    return result;
}

Tensor Tensor::operator * (const Tensor& other) const {
    Tensor result = copy();
    result *= other;
    return result;
}

Tensor Tensor::operator / (const Tensor& other) const {
    Tensor result = copy();
    result /= other;
    return result;
}

std::ostream& operator << (std::ostream& out, const Tensor& t) {
    const TensorParams& p = t.get_params();
    const Shape& shape = p.shape;
    size_t dims = shape.ndims();

    if (dims == 0) {
        return out << "[]";
    }

    lift(p.dtype, [&]<dtype_t tp>() {
        using Data = Info<tp>::Data;

        const Data* data = t.get_data<Data>();

        // Compute row-major strides
        std::vector<size_t> stride(dims);
        stride[dims - 1] = 1;
        for (int i = dims - 2; i >= 0; --i)
            stride[i] = stride[i + 1] * shape[i + 1];

        // Recursive ND print
        std::function<void(size_t, size_t)> rec =
            [&](size_t dim, size_t offset) {
            if (dim == dims - 1) {
                out << "[";
                for (size_t i = 0; i < shape[dim]; ++i) {
                    if (i) out << ", ";
                    out << data[offset + i];
                }
                out << "]";
                return;
            }

            out << "[";
            for (size_t i = 0; i < shape[dim]; ++i) {
                if (i) {
                    out << ",\n";
                    out << std::string(dim + 1, ' ');
                }
                rec(dim + 1, offset + i * stride[dim]);
            }
            out << "]";
            };

        rec(0, 0);
    });

    return out;
}
