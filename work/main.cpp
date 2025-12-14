#include <iostream>
// #include <torch/torch.h>

#include "tensor.h"
#include "graph/variable.h"
#include "tests/utils.h"

using namespace std;

void dprint(const Tensor& tensor) {
    std::cout << tensor.to(device_t::CPU) << std::endl;
}

int main() { 
    using Data = int;

    device_t device = device_t::GPU;

    Tensor x_t = Tensor::from_values(
        {1, 2, 3, 4}, 
        {.shape = {2, 2}, .device = device, .dtype = unlift<Data>}
    );

    Tensor y_t = Tensor::from_values(
        {1, 2, 3, 4}, 
        {.shape = {2, 2}, .device = device, .dtype = unlift<Data>}
    );

    VariableRef x = Variable::from(x_t);     // graph node
    VariableRef y = Variable::from(y_t);     // graph node

    VariableRef z = x->matmul(y);      // calculation graph

    VariableRef sum = z->sum();
    sum->backward();  // gradients calculation

    sum->cpu();

    cout << "Final value:" << endl;
    cout << sum->activation() << endl;

    cout << "X gradient:" << endl;
    cout << x->grad().to(device_t::CPU) << endl;

    cout << "Y gradient:" << endl;
    cout << y->grad().to(device_t::CPU) << endl;

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

