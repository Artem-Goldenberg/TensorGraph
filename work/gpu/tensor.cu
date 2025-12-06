#include "gpu/tensor.h"
#include "utils.h"

#include <sstream>

using namespace gpu;

namespace {

#define CHECK_CUDA_ERROR(val) cuda_check((val), #val, __FILE__, __LINE__)
void cuda_check(cudaError_t err, const char* const func, const char* const file,
    const int line) {
    if (err != cudaSuccess) {
        std::stringstream msg;
        msg << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        msg << cudaGetErrorString(err) << " " << func << std::endl;
        throw std::runtime_error(msg.str());
    }
}

// row-major, standard
template <typename Data>
__global__ void matmul_kernel(const Data* a, const Data* b, Data* out, long n, long l, long m) {
    // a: n x l
    // b: l x m
    // out: n x m

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n || j >= m) return;

    Data sum = 0;
    for (int k = 0; k < l; ++k)
        sum += a[i * l + k] * b[k * m + j];
    
    out[i * m + j] = sum;
}

template <typename Data, typename ScalarOp>
__global__ void bulk_kernel(Data* a, const Data* b, long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = ScalarOp::call(a[idx], b[idx]);
    }
}

template <typename T> struct Add { 
    __device__ static T call(T a, T b) { return a + b; }
};

template <typename T> struct Sub {
    __device__ static T call(T a, T b) { return a - b; }
};

template <typename T> struct Mult {
    __device__ static T call(T a, T b) { return a * b; }
};

template <typename T> struct Div {
    __device__ static T call(T a, T b) { return a / b; }
};

template <typename Data, typename Op>
float kernel_call(
    int grid_size, int block_size,
    Data* a, const Data* b, size_t n
) {
    cudaEvent_t start, stop;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    bulk_kernel<Data, Op> << <grid_size, block_size >> > (a, b, n);

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float milliseconds = 0;

    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return milliseconds;
}

template <dtype_t tp, template<typename> typename Op>
void bulk_operation(Tensor& a, const DeviceTensor& b) {
    using Data = typename Info<tp>::Data;

    check_bulk_compatible(a, b);

    auto params = a.get_params();

    kernel_call<Data, Op<Data>>(
        a.get_grid_size(), a.get_block_size(),
        a.get_mutable_data<Data>(),
        b.get_data<Data>(),
        params.shape.numel()
    );
}

}

Tensor::Tensor(const TensorParams& params, const void* data): DeviceTensor(params) {
    check(params.device == device_t::GPU, "Tried to instantiate a GPU Tensor with CPU parameters");

    size_t memsize = bytesize(params);

    void* gpu_data = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc(&gpu_data, memsize));

    assert(gpu_data);

    if (data) {
        cudaPointerAttributes attr{};
        cudaError_t err = cudaPointerGetAttributes(&attr, data);

        if (err == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
            // Source is GPU memory
            CHECK_CUDA_ERROR(cudaMemcpy(gpu_data, data, memsize, cudaMemcpyDeviceToDevice));
        } else {
            // Source is CPU memory
            CHECK_CUDA_ERROR(cudaMemcpy(gpu_data, data, memsize, cudaMemcpyHostToDevice));
        }
    }

    this->data = std::shared_ptr<void>(gpu_data, cudaFree);

    int max_threads_per_block = 0;

    CHECK_CUDA_ERROR(
        cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0)
    );

    this->block_size = std::min(max_threads_per_block, 1024);
    this->grid_size = (params.shape.numel() + block_size - 1) / block_size;
}

TensorRef Tensor::copy() const { 
    return std::make_shared<Tensor>(params, this->data.get());
}

TensorRef Tensor::matmul(const DeviceTensor& other) const {
    check_matmul_compatible(*this, other);

    size_t n = params.shape[0];
    size_t l = params.shape[1];
    size_t m = other.get_params().shape[1];

    // x is columns, y is rows
    dim3 block(16, 16);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    TensorRef result = std::make_shared<Tensor>(params.copy().with_shape({n, m}), nullptr);

    lift(params.dtype, [&]<dtype_t tp>() {
        using Data = Info<tp>::Data;

        Data* out = result->get_mutable_data<Data>();

        const Data* data = get_data<Data>();
        const Data* other_data = other.get_data<Data>();

        matmul_kernel<Data> <<<grid, block>>> (data, other_data, out, n, l, m);
    });

    return result;
}

void Tensor::add(const DeviceTensor& other) {
    lift(params.dtype, [&]<dtype_t tp>() {
        bulk_operation<tp, Add>(*this, other);
    });
}

void Tensor::subtract(const DeviceTensor& other) {
    lift(params.dtype, [&]<dtype_t tp>() {
        bulk_operation<tp, Sub>(*this, other);
    });
}

void Tensor::multiply(const DeviceTensor& other) {
    lift(params.dtype, [&]<dtype_t tp>() {
        bulk_operation<tp, Mult>(*this, other);
    });
}

void Tensor::divide(const DeviceTensor& other) {
    lift(params.dtype, [&]<dtype_t tp>() {
        bulk_operation<tp, Div>(*this, other);
    });
}

template <typename Data>
void Tensor::flush(Data *result) const {
    CHECK_CUDA_ERROR(cudaMemcpy(result, get_data<Data>(), bytesize(params), cudaMemcpyDeviceToHost));
}

#define InstantiateTemplates(tp, Data) \
    template void Tensor::flush<Data>(Data*) const;

ForEachDType(InstantiateTemplates)
