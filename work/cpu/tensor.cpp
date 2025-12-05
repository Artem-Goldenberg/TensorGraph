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
    return std::make_shared<Tensor>(params, this->data);
}

static void check_compatible(const Tensor& self, const DeviceTensor& other) {
    auto& params = self.get_params();
    auto& other_params = other.get_params();

    check(params.device == other_params.device,
        "Tried to do an operation on two tensors from different devices");
    
    check(params.dtype == other_params.dtype,
        "Cannot do an operation on tensors of different types");
}

static void check_bulk_compatible(const Tensor& self, const DeviceTensor& other) {
    check_compatible(self, other);

    check(self.get_params().shape == other.get_params().shape, 
        "Cannot do a bulk operation on a tenosrs of different shape");
}

template <dtype_t tp>
struct simd_foreach {
    using Data = Simd<tp>::Data;

    template <typename ScalarOp>
    static void call(
        Data* dst, const Data* src, size_t n,
        ScalarOp&& scalar_op, std::nullptr_t simd_op = nullptr
    ) {
        // Overload so that we can compute only with a scalar operation, when simd is not
        // available
        (void)simd_op;
        for (size_t i = 0; i < n; ++i)
            dst[i] = scalar_op(dst[i], src[i]);
    }

    template <typename ScalarOp, typename SimdOp>
    static void call(Data* dst, const Data* src, size_t n, ScalarOp&& scalar_op, SimdOp&& simd_op) {
        using Simd = Simd<tp>;
        size_t step = Simd::width;
        size_t simd_end = n / step * step; 
        for (size_t i = 0; i < simd_end; i += step) {
            auto reg1 = Simd::load(dst + i);
            auto reg2 = Simd::load(src + i);
            auto res = simd_op(reg1, reg2);
            Simd::store(dst + i, res);
        }
        call(dst + simd_end, src + simd_end, n - simd_end, scalar_op);
    }
};

template <dtype_t tp, typename ScalarOp, typename SimdOp>
static void bulk_operation(
    Tensor& a, const DeviceTensor& b,
    ScalarOp&& scalar_op, SimdOp&& simd_op
) {
    using Data = Simd<tp>::Data;

    check_bulk_compatible(a, b);

    auto params = a.get_params();

    simd_foreach<tp>::call(
        a.get_mutable_data<Data>(),
        b.get_data<Data>(),
        params.shape.numel(),
        scalar_op, simd_op
    );
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
        bulk_operation<tp>(*this, other, Scalar<tp>::add, nullptr);//, Simd<tp>::add);
    });
}

void Tensor::subtract(const DeviceTensor& other) {
    lift(params.dtype, [&]<dtype_t tp>() {
        bulk_operation<tp>(*this, other, Scalar<tp>::subtract, nullptr);//, Simd<tp>::subtract);
    });
}

void Tensor::multiply(const DeviceTensor& other) {
    lift(params.dtype, [&]<dtype_t tp>() {
        bulk_operation<tp>(*this, other, Scalar<tp>::multiply, nullptr);//, Simd<tp>::multiply);
    });
}

void Tensor::divide(const DeviceTensor& other) {
    lift(params.dtype, [&]<dtype_t tp>() {
        auto simd_div = Simd<tp>::divide;
        if constexpr (tp == dtype_t::Int32)
            simd_div = nullptr;  // no integer simd division
        bulk_operation<tp>(*this, other, Scalar<tp>::divide, nullptr);
    });
}
