#pragma once
#include <ostream>
#include <memory>
#include "device-tensor.h"

using std::shared_ptr;

class Tensor final {
public:
    template <typename Data>
    static Tensor from_blob(const Data* data, TensorParams params);

    template <typename Data>
    static Tensor from_values(std::initializer_list<Data> values, TensorParams params);

    static Tensor zeroes(TensorParams params);
    static Tensor ones(TensorParams params);

    template <typename Data>
    static Tensor fill(Data some, TensorParams params);

    const TensorParams& get_params() const;
    template <typename Data> const Data* get_data() const;
    template <typename Data> Data* get_mutable_data();

    Tensor copy() const;

    Tensor to(device_t device) const;

    void add(const Tensor& other);
    void subtract(const Tensor& other);
    void multiply(const Tensor& other);
    void divide(const Tensor& other);

    Tensor sum() const;

    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;

    Tensor& operator += (const Tensor& other);
    Tensor& operator -= (const Tensor& other);
    Tensor& operator *= (const Tensor& other);
    Tensor& operator /= (const Tensor& other);

    Tensor operator + (const Tensor& other) const;
    Tensor operator - (const Tensor& other) const;
    Tensor operator * (const Tensor& other) const;
    Tensor operator / (const Tensor& other) const;

private:
    shared_ptr<DeviceTensor> pImpl;

    Tensor(shared_ptr<DeviceTensor> pImpl): pImpl(pImpl) {}
};

std::ostream& operator<<(std::ostream& os, const Tensor& t);
