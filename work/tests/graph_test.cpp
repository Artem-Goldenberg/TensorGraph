#include "tensor.h"
#include "../utils.h"
#include "tests/utils.h"

#include "graph/variable.h"

#define ShapesMatMul \
    vector<size_t>{1, 1, 1}, \
    vector<size_t>{1, 1, 10}, \
    vector<size_t>{32, 16, 32}, \
    vector<size_t>{32, 32, 32}, \
    vector<size_t>{103, 101, 103}, \
    vector<size_t>{1024, 8, 8}, \
    vector<size_t>{1024, 200, 8}

template <typename Data>
static Tensor random_tensor(Shape shape, device_t device) {
    std::mt19937 rng(seed);
    vector<Data> vec = vector<Data>(shape.numel());

    fill_random_data<Data>(rng, vec.data(), shape.numel());

    return Tensor::from_blob(vec.data(), {.shape = shape, .device = device, .dtype = unlift<Data>});
}

template <typename Data>
static vector<VariableRef> linear_net(Tensor x, size_t l) {
    device_t device = x.get_params().device;

    Shape x_shape = x.get_params().shape;
    assert(x_shape.ndims() == 2);

    size_t n = x_shape[0];
    size_t m = x_shape[1];

    // y = x @ w + b
    // x: n x m
    // w: m x l
    // b: n x l
    Tensor w = random_tensor<Data>({m, l}, device);
    Tensor b = random_tensor<Data>({n, l}, device);

    VariableRef x_node = Variable::from(x);
    VariableRef w_node = Variable::from(w);
    VariableRef b_node = Variable::from(b);

    VariableRef mul = x_node->matmul(w_node);
    VariableRef result = *mul + b_node;

    VariableRef summed = result->sum();

    return {summed, w_node, b_node, x_node};
}

static void assert_all_on(const Variable& x, device_t device) {
    REQUIRE(x.get_params().device == device);
    for (VariableRef parent : x.get_parents())
        assert_all_on(*parent, device);
}

template <typename Data>
static void tensor_equals(Tensor a, Tensor b) {
    const Data* a_data = a.get_data<Data>();
    vector<Data> a_vec = vector(a_data, a_data + a.get_params().shape.numel());

    equals(a_vec, b.get_data<Data>());
}

// Test that gradient calculation on cpu and gpu gives the same results
TEMPLATE_TEST_CASE("grad calculation", "[grad]", int, float, double) { 
    using Data = TestType;

    vector<size_t> dim_pack = GENERATE(ShapesMatMul);

    size_t n = dim_pack[0];
    size_t m = dim_pack[1];
    size_t l = dim_pack[2];

    Tensor x_cpu = random_tensor<Data>({n, m}, device_t::CPU);
    Tensor x_gpu = random_tensor<Data>({n, m}, device_t::GPU);

    vector<VariableRef> cpu_nodes = linear_net<Data>(x_cpu, l);
    vector<VariableRef> gpu_nodes = linear_net<Data>(x_gpu, l);

    assert(cpu_nodes.size() == gpu_nodes.size());

    VariableRef cpu_result = cpu_nodes[0];
    VariableRef gpu_result = gpu_nodes[0];

    cpu_result->backward();
    gpu_result->backward();

    for (size_t i = 0; i < cpu_nodes.size(); ++i) {
        // Move to cpu to be able to retrieve results
        gpu_nodes[i]->cpu();

        // Compare gradients and activations
        tensor_equals<Data>(cpu_nodes[i]->grad(), gpu_nodes[i]->grad());
        tensor_equals<Data>(cpu_nodes[i]->activation(), gpu_nodes[i]->activation());
    }
}
