#include "tensor.h"
#include "../utils.h"
#include "tests/utils.h"

TEST_CASE("CPU copy", "[some]") {

    using Data = float;

    Shape shape = {101, 103};

    auto [a_data, b_data] = make_test_data<Data>(42, shape.numel());

    Tensor a = Tensor::from_blob<Data>(a_data, {.shape = shape, .dtype = unlift<Data>});
    // Tensor b = Tensor::from_blob(b_data, {.shape = shape, .dtype = dtype_t::Float32});

    Tensor b = a.copy();

    vector<Data> a_vec(a_data, a_data + shape.numel());

    equals(a_vec, b.get_data<Data>());

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


TEMPLATE_TEST_CASE(
    "addition", "[add]", int, float, double
) {
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

template <typename Data>
Matrix<Data> to_matrix(const Data* data, size_t rows, size_t cols) {
    Matrix<Data> mat(rows, vector<Data>(cols));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            mat[i][j] = data[i * cols + j];
    return mat;
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

    Data* a_data = new Data[a_shape.numel()];
    Data* b_data = new Data[b_shape.numel()];

    std::mt19937 rng(seed);

    fill_random_data<Data>(rng, a_data, a_shape.numel());
    fill_random_data<Data>(rng, b_data, b_shape.numel());

    Tensor a = Tensor::from_blob<Data>(
        a_data, {.shape = a_shape, .device = device, .dtype = unlift<Data>}
    );
    Tensor b = Tensor::from_blob<Data>(
        b_data, {.shape = b_shape, .device = device, .dtype = unlift<Data>}
    );

    Matrix<Data> a_mat = to_matrix(a_data, a_shape[0], a_shape[1]);
    Matrix<Data> b_mat = to_matrix(b_data, b_shape[0], b_shape[1]);

    Matrix<Data> result_mat = fullproof_matmul(a_mat, b_mat);

    // Flatten result back to row-major vector
    vector<Data> result(a_shape[0] * b_shape[1]);
    for (size_t i = 0; i < a_shape[0]; ++i)
        for (size_t j = 0; j < b_shape[1]; ++j)
            result[i * b_shape[1] + j] = result_mat[i][j];

    Tensor c = a.matmul(b);

    c = c.to(device_t::CPU);

    equals(result, c.get_data<Data>());

    delete[] a_data;
    delete[] b_data;

    // auto [a, b, result] = operation_test_data<Data>(dims, device, std::divides<Data>());

    // a /= b;
    // a = a.to(device_t::CPU);

    // equals(result, a.template get_data<Data>());
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
