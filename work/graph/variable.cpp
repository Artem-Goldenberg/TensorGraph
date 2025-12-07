#include "variable.h"

#include <set>
#include <algorithm>

using std::set;

Variable::Variable(Tensor value, vector<Variable*> parents) : 
    value(value), 
    gradient(Tensor::zeroes(value.get_params())),
    parents(parents),
    backward_pass([](auto){}) 
    {}

Tensor Variable::grad() const {
    return gradient;
}

const vector<Variable*>& Variable::get_parents() const { 
    return parents;
}

void Variable::cpu() {
    value = value.to(device_t::CPU);
    gradient = gradient.to(device_t::CPU);
    for (Variable* parent : parents) parent->cpu();
}

void Variable::gpu() {
    value = value.to(device_t::GPU);
    gradient = gradient.to(device_t::GPU);
    for (Variable* parent : parents) parent->gpu();
}

static void top_sort(Variable* current, set<Variable*>& visited, vector<Variable*>& sorted) { 
    if (visited.contains(current)) 
        return;

    visited.insert(current);
    for (Variable* parent : current->get_parents())
        top_sort(parent, visited, sorted);

    sorted.push_back(current);
}

void Variable::backward() {
    set<Variable*> visited = {};
    vector<Variable*> sorted = {};

    top_sort(this, visited, sorted);
    std::reverse(sorted.begin(), sorted.end());

    for (Variable* node : sorted)
        node->backward_pass(*node);
}

Variable Variable::operator + (Variable& other) {
    Variable result = Variable(value + other.value, {this, &other});

    result.backward_pass = [](Variable& result) {
        result.parents[0]->gradient += result.gradient; 
        result.parents[1]->gradient += result.gradient;
    };

    return result;
}

Variable Variable::operator - (Variable& other) {
    Variable result = Variable(value - other.value, {this, &other});

    result.backward_pass = [](Variable& result) {
        result.parents[0]->gradient += result.gradient;
        result.parents[1]->gradient -= result.gradient;
    };

    return result;
}

Variable Variable::operator * (Variable& other) {
    Variable result = Variable(value * other.value, {this, &other});

    result.backward_pass = [](Variable& result) {
        Variable& a = *result.parents[0];
        Variable& b = *result.parents[1];
        a.gradient += b.value * result.gradient;
        b.gradient += a.value * result.gradient;
    };

    return result;
}

Variable Variable::operator / (Variable& other) {
    Variable result = Variable(value / other.value, {this, &other});

    result.backward_pass = [](Variable& result) {
        Variable& a = *result.parents[0];
        Variable& b = *result.parents[1];

        // a / b = (a * b**-1)
        // d(a * b**-1)/da = b**-1
        // d(a * b**-1)/db = -a * b**-2

        Tensor one_b = Tensor::ones(b.value.get_params());

        a.gradient += one_b / b.value * result.gradient;
        b.gradient -= a.value / (b.value * b.value) * result.gradient;
    };

    return result;
}

Variable Variable::matmul(Variable& other) {
    Variable result = Variable(value.matmul(other.value), {this, &other});

    result.backward_pass = [](Variable& result) {
        Variable& a = *result.parents[0];
        Variable& b = *result.parents[1];
        // a, da: n x l
        // b, db: l x m
        // a @ b, grad: n x m
        // da = grad @ b.T
        // db = a.T @ grad
        a.gradient += result.gradient.matmul(b.value.transpose());
        b.gradient += a.value.transpose().matmul(result.gradient);
    };

    return result;
}

Variable Variable::sum() {
    Variable result = Variable(value.sum(), {this});
    result.backward_pass = [](Variable& result) {
        const TensorParams& params = result.gradient.get_params();
        const Shape& shape = params.shape;

        check(shape.ndims() == 1 && shape[0] == 1, "Sum gradient must be a signle scalar");

        lift(params.dtype, [&]<dtype_t tp>() {
            using Data = Info<tp>::Data;
            const Data* grad_data = result.gradient.get_data<Data>();

            Tensor& parent_grad = result.parents[0]->gradient;

            // Broadcast the scalar value to the parent tensor scale 
            parent_grad += Tensor::from_values({grad_data[0]}, parent_grad.get_params());
        });
    };
    return result;
}
