#include <iostream>
// #include <torch/torch.h>

#include "tensor.h"
#include "graph/variable.h"
#include "tests/utils.h"

using namespace std;


int main() { 
    using Data = int;

    size_t m = 16;
    size_t n = 32;

    // float* data_x{nullptr};
    // float* data_y{nullptr};

    auto [data_x, data_y] = make_test_data<Data>(107, n * m);

    Tensor x_t = Tensor::from_blob(
        data_x, { .shape = {m, n}, .device = device_t::GPU, .dtype = unlift<Data> }
    );

    Tensor y_t = Tensor::from_blob(
        data_y, { .shape = {m, n}, .device = device_t::GPU, .dtype = unlift<Data> }
    );
    
    Variable x(x_t);     // graph node
    Variable y(y_t);     // graph node

    Variable z = x * y;      // calculation graph

    z.cpu();             // if there is no sum implementation for GPU
    z.sum().backward();  // gradients calculation

    cout << x.grad() << endl;
    cout << y.grad() << endl;

    // vector<Data>

    // Compare with LibTorch
    // torch::Tensor x_torch = torch::from_blob(data_x, { (long)m, (long)n }, torch::kFloat32);
    // torch::Tensor y_torch = torch::from_blob(data_y, { (long)m, (long)n }, torch::kFloat32);

    // torch::Tensor z_torch = x_torch * y_torch;
    // torch::Tensor grad = torch::ones_like(z_torch); // gradient for sum

    // z_torch.backward(grad);

    // std::cout << "LibTorch gradients:\n";
    // std::cout << x_torch.grad() << std::endl;

    return 0;
}

