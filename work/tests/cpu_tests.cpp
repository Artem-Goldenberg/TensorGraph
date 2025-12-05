#include "tensor.h"
#include "../utils.h"
#include "tests/utils.h"

#define TEST_VARIANTS \
    ((size_t N), N), \
    (Int32Case, Float32Case, Float64Case), \
    ((8), (32), (512), (4096), (1024 * 1024))

// #define TEST_VARIANTS \
//     ((size_t N), N), \
//     (Float32Case), \
//     ((8))

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
    vector<size_t> dims, ScalarOp&& scalar_op
) {
    Shape shape(dims);

    auto [a_data, b_data] = make_test_data<Data>(seed, shape.numel());

    Tensor a = Tensor::from_blob(a_data, {.shape = shape, .dtype = unlift<Data>});
    Tensor b = Tensor::from_blob(b_data, {.shape = shape, .dtype = unlift<Data>});

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
    

TEMPLATE_TEST_CASE(
    "CPU addition", "[add]", int, float, double
) {
    using Data = TestType;
    vector<size_t> dims = GENERATE(Shapes);
    auto [a, b, result] = operation_test_data<Data>(dims, std::plus<Data>());
    a += b;
    equals(result, a.template get_data<Data>());
}

TEMPLATE_TEST_CASE(
    "CPU subtraction", "[sub]", int, float, double
) {
    using Data = TestType;
    vector<size_t> dims = GENERATE(Shapes);
    auto [a, b, result] = operation_test_data<Data>(dims, std::minus<Data>());
    a -= b;
    equals(result, a.template get_data<Data>());
}

TEMPLATE_TEST_CASE(
    "CPU multiplication", "[mult]", int, float, double
) {
    using Data = TestType;
    vector<size_t> dims = GENERATE(Shapes);
    auto [a, b, result] = operation_test_data<Data>(dims, std::multiplies<Data>());
    a *= b;
    equals(result, a.template get_data<Data>());
}

TEMPLATE_TEST_CASE(
    "CPU division", "[div]", float, double
) {
    using Data = TestType;
    vector<size_t> dims = GENERATE(Shapes);
    auto [a, b, result] = operation_test_data<Data>(dims, std::divides<Data>());
    a /= b;
    equals(result, a.template get_data<Data>());
}


// TEMPLATE_PRODUCT_TEST_CASE_SIG(
//     "CPU add",
//     "[arithmetices]",
//     TEST_VARIANTS
// ) {
//     using Data = TestType::T;

//     Shape shape = {101, 103};

//     vector<Data> a_vec(a_data, a_data + shape.numel());

//     for (size_t i = 0; i < a_vec.size(); ++i) {
//         // printf("%d\n", i);
//         a_vec[i] += b_data[i];
//     }

//     a += b;

//     equals(a_vec, a.get_data<Data>());

    // std::vector<typename TestType::T> result(TestType::N);
    // cpu::add_vec(a.data(), b.data(), result.data(), TestType::N);

    // std::vector<typename TestType::T> expected(TestType::N);
    // baseline::add_vec(a.data(), b.data(), expected.data(), TestType::N);
    // equals(expected, result.data());

    // Example how to add perf benchmark to the Catch2 tests
    // BENCHMARK("add") {
    //   cpu::add_vec(a.data(), b.data(), result.data(), TestType::N);
    // };
// }
