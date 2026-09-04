#include "cpu/tensor.h"

#include "simd.h"
#include "utils.h"
#include "device-tensor.h"
#include "tensor.h"
#include "omp.h"

using namespace cpu;

constexpr static size_t parallelization_threshold = 100;

template <typename Data>
struct dtcopy {
    static void call(Data* dst, const Data* src, size_t n) {
        using Simd = Simd<unlift<Data>>;
        using simd = Simd::simd;

        constexpr size_t step = Simd::width;
        size_t simd_end = (n / step) * step;

        #pragma omp parallel for if (simd_end > parallelization_threshold)
        for (size_t i = 0; i < simd_end / step; ++i) {
            size_t offset = i * step;

            simd reg = Simd::load(src + offset);
            Simd::store(dst + offset, reg);
        }

        std::copy(src + simd_end, src + n, dst + simd_end);
    }

    static void set(Data* dst, Data eleme, size_t n) {
        using Simd = Simd<unlift<Data>>;
        using simd = Simd::simd;

        constexpr size_t step = Simd::width;
        size_t simd_end = (n / step) * step;

        simd elem_reg = Simd::set(eleme);

        #pragma omp parallel for if (simd_end > parallelization_threshold)
        for (size_t i = 0; i < simd_end / step; ++i) {
            size_t offset = i * step;
            Simd::store(dst + offset, elem_reg);
        }

        std::fill(dst + simd_end, dst + n, eleme);
    }
};

static std::shared_ptr<void> allocate_aligned_memory(const TensorParams& params) {
    void* raw = std::aligned_alloc(32, bytesize(params));
    assert(raw);
    return std::shared_ptr<void>(raw, std::free);
}

Tensor::Tensor(const TensorParams& params, const void* data): DeviceTensor(params) {
    check(params.device == device_t::CPU, "Tried to instantiate a CPU Tensor with GPU parameters");

    size_t n = params.shape.numel();
    this->data = allocate_aligned_memory(params);

    if (data)
        lift(params.dtype, [&]<dtype_t tp>() {
            using Data = Info<tp>::Data;
            dtcopy<Data>::call((Data*)this->data.get(), (const Data*)data, n);
        });
}

template<typename Data>
Tensor::Tensor(Data elem, const TensorParams& params): DeviceTensor(params) {
    check(params.device == device_t::CPU, "Tried to instantiate a CPU Tensor with GPU parameters");

    size_t n = params.shape.numel();
    this->data = allocate_aligned_memory(params);

    dtcopy<Data>::set((Data*)this->data.get(), elem, n);
}

// Instantiate the constuctor above for all the data types needed
#define InstantiateTemplates(tp, Data) \
    template Tensor::Tensor<Data>(Data, const TensorParams&);
ForEachDType(InstantiateTemplates)

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

        // simd sum = Simd::load(buffer);
        simd sum = Simd::set(0);

        size_t n = params.shape.numel();
        size_t simd_end = n / step * step;

        // const Data* result->get_data<Data>();
        const Data* data = get_data<Data>();

        for (size_t i = 0; i < simd_end / step; ++i) {
            size_t offset = i * step;
            simd v = Simd::load(data + offset);
            sum = Simd::add(sum, v);
        }

        // add the 8 elements manually
        alignas(32) Data buffer[8] = {};
        Simd::store(buffer, sum);

        Data total = 0;
        for (size_t i = 0; i < step; ++i)
            total += buffer[i];

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

        #pragma omp parallel for collapse(2) if (n * m > parallelization_threshold)
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

        #pragma omp parallel for collapse(2) if (n * m > parallelization_threshold)
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
        #pragma omp parallel for if (simd_end > parallelization_threshold)
        for (size_t i = 0; i < simd_end / step; ++i) {
            size_t offset = i * step;
            auto reg1 = Simd::load(dst + offset);
            auto reg2 = Simd::load(src + offset);
            auto res = simd_op(reg1, reg2);
            Simd::store(dst + offset, res);
        }
    }

    #pragma omp parallel for if (n - simd_end > parallelization_threshold)
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

void Tensor::clear() {
    lift(params.dtype, [&]<dtype_t tp>() {
        using Data = Info<tp>::Data;

        Data* dst = get_mutable_data<Data>();

        dtcopy<Data>::set(dst, (Data)0, params.shape.numel());
    });
}
