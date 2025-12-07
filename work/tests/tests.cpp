#include "tensor.h"
#include "../utils.h"
#include "tests/utils.h"

template <typename Data>
static std::tuple<Tensor, Matrix<Data>> matrix_test_data(const Shape& shape, device_t device) {
    Data* data = new Data[shape.numel()];

    std::mt19937 rng(seed);

    fill_random_data<Data>(rng, data, shape.numel());

    Tensor a = Tensor::from_blob<Data>(
        data, {.shape = shape, .device = device, .dtype = unlift<Data>}
    );

    Matrix<Data> mat = to_matrix(data, shape[0], shape[1]);

    delete[] data;

    return {a, mat};
}

template <typename Data, typename ScalarOp>
static std::tuple<Tensor, Tensor, vector<Data>> operation_test_data(
    vector<size_t> dims, device_t device, ScalarOp&& scalar_op
) {
    Shape shape(dims);

    auto [a_data, b_data] = make_test_data<Data>(seed, shape.numel());

    Tensor a = Tensor::from_blob(a_data, {.shape = shape, .device = device, .dtype = unlift<Data>});
    Tensor b = Tensor::from_blob(b_data, {.shape = shape, .device = device, .dtype = unlift<Data>});

    vector<Data> a_vec(a_data, a_data + shape.numel());

    for (size_t i = 0; i < a_vec.size(); ++i)
        a_vec[i] = scalar_op(a_vec[i], b_data[i]);

    delete[] a_data;
    delete[] b_data;

    return {a, b, a_vec};
}

#define Shapes \
    vector<size_t>{1}, \
    vector<size_t>{1, 1}, \
    vector<size_t>{1, 1, 1}, \
    vector<size_t>{32, 32}, \
    vector<size_t>{103, 101}, \
    vector<size_t>{1024, 200, 8}

// In the format: (n, l, m) for (n, l) x (l, m) --> (n, m)
#define ShapesMatMul \
    vector<size_t>{1, 1, 1}, \
    vector<size_t>{1, 1, 10}, \
    vector<size_t>{32, 16, 32}, \
    vector<size_t>{32, 32, 32}, \
    vector<size_t>{103, 101, 103}, \
    vector<size_t>{1024, 8, 8}, \
    vector<size_t>{1024, 200, 8}
    // vector<size_t>{1024, 1024, 1024}

#define Shapes2D \
    vector<size_t>{1, 1}, \
    vector<size_t>{1, 2}, \
    vector<size_t>{2, 1}, \
    vector<size_t>{16, 15}, \
    vector<size_t>{101, 103}, \
    vector<size_t>{1024, 1024}


TEMPLATE_TEST_CASE("copying", "[copy]", int, float, double) {
    using Data = TestType;

    device_t device = GENERATE(device_t::CPU, device_t::GPU);
    Shape shape = GENERATE(Shapes);

    auto [data, to_free] = make_test_data<Data>(seed, shape.numel());

    Tensor a = Tensor::from_blob<Data>(
        data, {.shape = shape, .device = device, .dtype = unlift<Data>}
    );

    Tensor b = a.copy();
    b = b.to(device_t::CPU);

    vector<Data> vec = vector<Data>(data, data + shape.numel());
    equals(vec, b.get_data<Data>());

    delete[] data;
    delete[] to_free;
}

TEMPLATE_TEST_CASE("addition", "[add]", int, float, double) {
    using Data = TestType;

    vector<size_t> dims = GENERATE(Shapes);
    device_t device = GENERATE(device_t::CPU, device_t::GPU);

    CAPTURE(dims);
    CAPTURE(device);

    auto [a, b, result] = operation_test_data<Data>(dims, device, std::plus<Data>());

    a += b;
    a = a.to(device_t::CPU);

    equals(result, a.template get_data<Data>());
}

TEMPLATE_TEST_CASE(
    "subtraction", "[sub]", int, float, double
) {
    using Data = TestType;

    vector<size_t> dims = GENERATE(Shapes);
    device_t device = GENERATE(device_t::CPU, device_t::GPU);

    CAPTURE(dims);
    CAPTURE(device);

    auto [a, b, result] = operation_test_data<Data>(dims, device, std::minus<Data>());

    a -= b;
    a = a.to(device_t::CPU);

    equals(result, a.template get_data<Data>());
}

TEMPLATE_TEST_CASE(
    "multiplication", "[mult]", int, float, double
) {
    using Data = TestType;

    vector<size_t> dims = GENERATE(Shapes);
    device_t device = GENERATE(device_t::CPU, device_t::GPU);

    CAPTURE(dims);
    CAPTURE(device);

    auto [a, b, result] = operation_test_data<Data>(dims, device, std::multiplies<Data>());

    a *= b;
    a = a.to(device_t::CPU);

    equals(result, a.template get_data<Data>());
}

TEMPLATE_TEST_CASE(
    "division", "[div]", int, float, double
) {
    using Data = TestType;

    vector<size_t> dims = GENERATE(Shapes);
    device_t device = GENERATE(device_t::CPU, device_t::GPU);

    CAPTURE(dims);
    CAPTURE(device);

    auto [a, b, result] = operation_test_data<Data>(dims, device, std::divides<Data>());

    a /= b;
    a = a.to(device_t::CPU);

    equals(result, a.template get_data<Data>());
}

TEMPLATE_TEST_CASE("sum", "[sum]", int, float, double) { 
    using Data = TestType;

    Shape shape = GENERATE(Shapes);
    device_t device = GENERATE(device_t::CPU, device_t::GPU);

    CAPTURE(shape.get_dims());
    CAPTURE(device);

    Data* data = new Data[shape.numel()];

    std::mt19937 rng(seed);

    fill_random_data<Data>(rng, data, shape.numel());

    Tensor a = Tensor::from_blob<Data>(
        data, {.shape = shape, .device = device, .dtype = unlift<Data>}
    );

    Data sum = 0;
    for (uint i = 0; i < shape.numel(); ++i) sum += data[i];

    Tensor single = a.sum().to(device_t::CPU);

    Data result = single.get_data<Data>()[0];

    REQUIRE(nearly_equal_impl(result, sum));

    delete[] data;
}

TEMPLATE_TEST_CASE("transpose", "[tp]", int, float, double) {
    using Data = TestType;

    vector<size_t> dims = GENERATE(Shapes2D);

    device_t device = GENERATE(device_t::CPU, device_t::GPU);

    CAPTURE(dims);
    CAPTURE(device);

    Shape shape = Shape(dims);

    REQUIRE(shape.ndims() == 2);

    auto [a, mat] = matrix_test_data<Data>(shape, device);

    Matrix<Data> result_mat = fullproof_transpose(mat);
    vector<Data> result = from_matrix(result_mat);

    Tensor b = a.transpose();
    b = b.to(device_t::CPU);

    equals(result, b.get_data<Data>());
}

TEMPLATE_TEST_CASE(
    "matrix multiplication", "[matmul]", int, float, double
) {
    using Data = TestType;

    vector<size_t> dims_pack = GENERATE(ShapesMatMul);

    device_t device = GENERATE(device_t::CPU, device_t::GPU);

    CAPTURE(dims_pack);
    CAPTURE(device);

    Shape a_shape = vector<size_t>(dims_pack.begin(), dims_pack.begin() + 2);
    Shape b_shape = vector<size_t>(dims_pack.begin() + 1, dims_pack.end());

    REQUIRE(a_shape.ndims() == 2);
    REQUIRE(b_shape.ndims() == 2);
    REQUIRE(a_shape[1] == b_shape[0]);

    auto [a, a_mat] = matrix_test_data<Data>(a_shape, device);
    auto [b, b_mat] = matrix_test_data<Data>(b_shape, device);

    Matrix<Data> result_mat = fullproof_matmul(a_mat, b_mat);
    vector<Data> result = from_matrix(result_mat);

    Tensor c = a.matmul(b);
    c = c.to(device_t::CPU);

    equals(result, c.get_data<Data>());
}


// // TEMPLATE_PRODUCT_TEST_CASE_SIG(
// //     "CPU add",
// //     "[arithmetices]",
// //     TEST_VARIANTS
// // ) {
// //     using Data = TestType::T;

// //     Shape shape = {101, 103};

// //     vector<Data> a_vec(a_data, a_data + shape.numel());

// //     for (size_t i = 0; i < a_vec.size(); ++i) {
// //         // printf("%d\n", i);
// //         a_vec[i] += b_data[i];
// //     }

// //     a += b;

// //     equals(a_vec, a.get_data<Data>());

//     // std::vector<typename TestType::T> result(TestType::N);
//     // cpu::add_vec(a.data(), b.data(), result.data(), TestType::N);

//     // std::vector<typename TestType::T> expected(TestType::N);
//     // baseline::add_vec(a.data(), b.data(), expected.data(), TestType::N);
//     // equals(expected, result.data());

//     // Example how to add perf benchmark to the Catch2 tests
//     // BENCHMARK("add") {
//     //   cpu::add_vec(a.data(), b.data(), result.data(), TestType::N);
//     // };
// // }
