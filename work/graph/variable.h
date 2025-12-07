#include "tensor.h"

#include <functional>

class Variable;

using std::vector;
// using VariableRef = std::shared_ptr<Variable>;

class Variable {
    Tensor value;
    Tensor gradient;

    vector<Variable*> parents;

    // Call it like a method: pass `this`
    std::function<void(Variable& self)> backward_pass;

public:
    Variable(Tensor value, vector<Variable*> = {});

    Tensor grad() const;
    const vector<Variable*>& get_parents() const;

    void backward(); // Do a backward pass from this node

    Variable operator + (Variable& other);
    Variable operator - (Variable& other);
    Variable operator * (Variable& other);
    Variable operator / (Variable& other);

    Variable matmul(Variable& other);
    Variable sum();

    // Switch computations to cpu
    void cpu();
    // Switch computations to gpu
    void gpu();
};
