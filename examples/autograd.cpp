// Builds a tiny computation graph on the GPU, runs a backward pass, and
// prints the result and the gradients.
#include <iostream>

#include "tensor.h"
#include "utils.h"
#include "graph/variable.h"

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

    return 0;
}

