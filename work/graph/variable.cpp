#include "variable.h"
#include <iostream>

#include <set>
#include <algorithm>

using std::set;
using std::make_shared;

using Parents = vector<VariableRef>;

Variable::Variable(Tensor value, vector<VariableRef> parents) : 
    value(value), 
    gradient(Tensor::zeroes(value.get_params())),
    parents(parents),
    backward_pass([](auto, auto){}) 
    {}

Tensor Variable::activation() const {
    return value;
}

VariableRef Variable::from(Tensor value) {
    return make_shared<Variable>(value);
}

Tensor Variable::grad() const {
    return gradient;
}

const vector<VariableRef>& Variable::get_parents() const { 
    return parents;
}

const TensorParams& Variable::get_params() const {
    return value.get_params();
}

void Variable::cpu() {
    value = value.to(device_t::CPU);
    gradient = gradient.to(device_t::CPU);
}

void Variable::gpu() {
    value = value.to(device_t::GPU);
    gradient = gradient.to(device_t::GPU);
}

void Variable::clear() {
    gradient.clear();
    for (VariableRef parent : parents) parent->clear();
}

static void top_sort(VariableRef current, set<VariableRef> &visited, vector<VariableRef> &sorted) { 
    if (visited.contains(current)) 
        return;

    visited.insert(current);
    for (VariableRef parent : current->get_parents())
        top_sort(parent, visited, sorted);

    sorted.push_back(current);
}

// Asserts the provided nodes are all from the same device and returns this device
static device_t get_common_device(const vector<VariableRef> &nodes) {
    assert(!nodes.empty());

    device_t common_device = nodes[0]->activation().get_params().device;

    for (VariableRef node : nodes) { 
        device_t grad_device = node->grad().get_params().device;
        device_t value_device = node->activation().get_params().device;

        check(value_device == grad_device, 
            "Node has a mismatching devices for it's gradient and value");

        check(value_device == common_device,
            "Parent nodes have a mismatching devices");
    }

    return common_device;
}

void Variable::backward() {
    set<VariableRef> visited = {};
    vector<VariableRef> sorted = {};

    top_sort(shared_from_this(), visited, sorted);
    std::reverse(sorted.begin(), sorted.end());

    if (sorted.empty()) return;

    sorted[0]->gradient = Tensor::ones(sorted[0]->value.get_params());

    for (VariableRef node : sorted) {
        const vector<VariableRef> &parents = node->get_parents();

        if (parents.empty()) 
            // Don't need a backward pass if don't have parents
            continue;

        device_t device = get_common_device(parents);

        // Transfer the gradient to the device needed for the backward computation
        node->backward_pass(*node, node->gradient.to(device));
    }
}

VariableRef Variable::operator + (VariableRef other) {
    VariableRef result = make_shared<Variable>(
        value + other->value,
        Parents {shared_from_this(), other}
    );

    result->backward_pass = [](Variable& result, Tensor gradient) {
        result.parents[0]->gradient += gradient; 
        result.parents[1]->gradient += gradient;
    };

    return result;
}

VariableRef Variable::operator - (VariableRef other) {
    VariableRef result = make_shared<Variable>(
        value - other->value,
        Parents {shared_from_this(), other}
    );

    result->backward_pass = [](Variable& result, Tensor gradient) {
        result.parents[0]->gradient += gradient;
        result.parents[1]->gradient -= gradient;
    };

    return result;
}

VariableRef Variable::operator * (VariableRef other) {
    VariableRef result = make_shared<Variable>(
        value * other->value,
        Parents {shared_from_this(), other}
    );

    result->backward_pass = [](Variable& result, Tensor gradient) {
        Variable& a = *result.parents[0];
        Variable& b = *result.parents[1];
        a.gradient += b.value * gradient;
        b.gradient += a.value * gradient;
    };

    return result;
}

VariableRef Variable::operator / (VariableRef other) {
    VariableRef result = make_shared<Variable>(
        value / other->value,
        Parents {shared_from_this(), other}
    );

    result->backward_pass = [](Variable& result, Tensor gradient) {
        Variable& a = *result.parents[0];
        Variable& b = *result.parents[1];

        // a / b = (a * b**-1)
        // d(a * b**-1)/da = b**-1
        // d(a * b**-1)/db = -a * b**-2

        Tensor one_b = Tensor::ones(b.value.get_params());

        a.gradient += one_b / b.value * gradient;
        b.gradient -= a.value / (b.value * b.value) * gradient;
    };

    return result;
}

VariableRef Variable::matmul(VariableRef other) {
    VariableRef result = make_shared<Variable>(
        value.matmul(other->value),
        Parents {shared_from_this(), other}
    );

    result->backward_pass = [](Variable& result, Tensor gradient) {
        Variable& a = *result.parents[0];
        Variable& b = *result.parents[1];
        // a, da: n x l
        // b, db: l x m
        // a @ b, grad: n x m
        // da = grad @ b.T
        // db = a.T @ grad
        a.gradient += gradient.matmul(b.value.transpose());
        b.gradient += a.value.transpose().matmul(gradient);
    };

    return result;
}

VariableRef Variable::sum() {
    VariableRef result = make_shared<Variable>(value.sum(), Parents {shared_from_this()});
    result->backward_pass = [](Variable& result, Tensor gradient) {
        const TensorParams& params = gradient.get_params();
        const Shape& shape = params.shape;

        check(shape.ndims() == 1 && shape[0] == 1, "Sum gradient must be a signle scalar");

        lift(params.dtype, [&]<dtype_t tp>() {
            using Data = Info<tp>::Data;
            // Convert to CPU in case the computation of the sum was left on the GPU
            Tensor cpu_grad = gradient.to(device_t::CPU);
            Data value = cpu_grad.get_data<Data>()[0];

            Tensor& parent_grad = result.parents[0]->gradient;

            // Broadcast the scalar value to the parent tensor scale 
            parent_grad += Tensor::fill(value, parent_grad.get_params());
        });
    };
    return result;
}
