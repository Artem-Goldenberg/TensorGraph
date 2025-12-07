#include "cpu/tensor.h"

#include "simd.h"
#include "utils.h"
#include "device-tensor.h"
#include "tensor.h"

using namespace cpu;

template<dtype_t tp>
struct dtcopy {
    using Data = Simd<tp>::Data;

    static void call(Data* dst, const Data* src, size_t n) {
        std::copy(src, src + n, dst);
        // for (size_t i = 0; i < n; i += Simd<tp>::width) {
        //     auto reg = Simd<tp>::load(src + i);
        //     Simd<tp>::store(dst + i, reg);
        // }
    }
};

Tensor::Tensor(const TensorParams& params, const void* data): DeviceTensor(params) {
    check(params.device == device_t::CPU, "Tried to instantiate a CPU Tensor with GPU parameters");

    size_t n = params.shape.numel();
    void* raw = std::aligned_alloc(32, n * size_of(params.dtype));

    assert(raw);

    this->data = std::shared_ptr<void>(raw, std::free);

    if (data)
        lift(params.dtype, [&]<dtype_t tp>() {
            using Data = Simd<tp>::Data;
            dtcopy<tp>::call((Data*)this->data.get(), (const Data*)data, n);
        });
}

TensorRef Tensor::copy() const {
    assert(this->data);
    return std::make_shared<Tensor>(params, this->data.get());
}

TensorRef Tensor::sum() const {
    validate(params.shape);

    TensorRef result = std::make_shared<Tensor>(params.copy().with_shape({1}), nullptr);

    lift(params.dtype, [&]<dtype_t tp>() { 
        using Simd = Simd<tp>;
        using Data = Simd::Data;
        using simd = Simd::simd; // register type

        static const size_t step = Simd::width;

        alignas(32) Data buffer[8] = {}; // zeroes

        simd sum = Simd::load(buffer);

        size_t n = params.shape.numel();
        size_t simd_end = n / step * step;

        // const Data* result->get_data<Data>();
        const Data* data = get_data<Data>();

        for (size_t i = 0; i < simd_end; i += step) {
            simd v = Simd::load(data + i);
            sum = Simd::add(sum, v);
        }

        // store in buffer and add manually
        Simd::store(buffer, sum);
        Data total = buffer[0] + buffer[1] + buffer[2] + buffer[3] +
                    buffer[4] + buffer[5] + buffer[6] + buffer[7];

        // remaining elements
        for (size_t i = simd_end; i < n; ++i)
            total += data[i];
        
        Data* out = result->get_mutable_data<Data>();
        out[0] = total;
    });

    return result;
}

TensorRef Tensor::matmul(const DeviceTensor& other) const {
    check_matmul_compatible(*this, other);

    // n, l * l, m --> n, m

    size_t n = params.shape[0];
    size_t l = params.shape[1];
    size_t m = other.get_params().shape[1];

    TensorRef result = std::make_shared<Tensor>(params.copy().with_shape({n, m}), nullptr);

    lift(params.dtype, [&]<dtype_t tp>() { 
        using Data = Info<tp>::Data;

        const Data* data = get_data<Data>();
        const Data* other_data = other.get_data<Data>();

        Data* result_data = result->get_mutable_data<Data>();

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                Data sum = 0;
                for (size_t k = 0; k < l; ++k) {
                    sum += data[i * l + k] * other_data[k * m + j];
                }
                result_data[i * m + j] = sum;
            }
        }
    });

    return result;
}

TensorRef Tensor::transpose() const {
    check_transposable(*this);

    size_t n = params.shape[0];
    size_t m = params.shape[1];

    TensorRef result = std::make_shared<Tensor>(params.copy().with_shape({m, n}), nullptr);

    lift(params.dtype, [&]<dtype_t tp>() { 
        using Data = Info<tp>::Data;

        const Data* data = this->get_data<Data>();
        Data* result_data = result->get_mutable_data<Data>();

        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < m; ++j)
                result_data[j * n + i] = data[i * m + j];
    });

    return result;
}

template <dtype_t tp, typename ScalarOp, typename SimdOp = nullptr_t>
static void bulk_operation(
    Tensor& a, const DeviceTensor& b,
    ScalarOp&& scalar_op, SimdOp&& simd_op = nullptr
) {
    using Simd = Simd<tp>;
    using Data = Simd::Data;

    check_bulk_compatible(a, b);

    Data* dst = a.get_mutable_data<Data>();
    const Data* src = b.get_data<Data>();

    size_t n = a.get_params().shape.numel();

    size_t step = Simd::width;
    size_t simd_end = 0;

    if constexpr (!std::is_same_v<SimdOp, nullptr_t>) {
        simd_end = n / step * step; 
        for (size_t i = 0; i < simd_end; i += step) {
            auto reg1 = Simd::load(dst + i);
            auto reg2 = Simd::load(src + i);
            auto res = simd_op(reg1, reg2);
            Simd::store(dst + i, res);
        }
    }

    for (size_t i = simd_end; i < n; ++i)
        dst[i] = scalar_op(dst[i], src[i]);
}

template <dtype_t tp>
struct Scalar {
    using Data = typename Simd<tp>::Data;

    static Data add(Data a, Data b) { return a + b; }
    static Data subtract(Data a, Data b) { return a - b; }
    static Data multiply(Data a, Data b) { return a * b; }
    static Data divide(Data a, Data b) { return a / b; }
};

void Tensor::add(const DeviceTensor& other) {
    lift(params.dtype, [&]<dtype_t tp>() {
        bulk_operation<tp>(*this, other, Scalar<tp>::add, Simd<tp>::add);
    });
}

void Tensor::subtract(const DeviceTensor& other) {
    lift(params.dtype, [&]<dtype_t tp>() {
        bulk_operation<tp>(*this, other, Scalar<tp>::subtract, Simd<tp>::subtract);
    });
}

void Tensor::multiply(const DeviceTensor& other) {
    lift(params.dtype, [&]<dtype_t tp>() {
        bulk_operation<tp>(*this, other, Scalar<tp>::multiply, Simd<tp>::multiply);
    });
}

void Tensor::divide(const DeviceTensor& other) {
    lift(params.dtype, [&]<dtype_t tp>() {
        if constexpr (tp == dtype_t::Int32)
            // no integer simd division
            bulk_operation<tp>(*this, other, Scalar<tp>::divide, nullptr);
        else
            bulk_operation<tp>(*this, other, Scalar<tp>::divide, Simd<tp>::divide);
    });
}
