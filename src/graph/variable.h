#include "tensor.h"

#include <functional>

class Variable;

using std::vector;
using VariableRef = std::shared_ptr<Variable>;

class Variable: public std::enable_shared_from_this<Variable> {
    Tensor value;
    Tensor gradient;

    vector<VariableRef> parents;

    // Pass `this` node for the self argument, also pass the appropriate gradient
    std::function<void(Variable& self, Tensor gradient)> backward_pass;

public:
    Variable(Tensor value, vector<VariableRef> = {});

    static VariableRef from(Tensor value);

    Tensor grad() const;
    Tensor activation() const;
    const vector<VariableRef>& get_parents() const;

    const TensorParams& get_params() const;

    // Do a backward pass from this node
    // If for some reason you're calling this the second time on the same node,
    // then don't forget to clear out the gradients beforehand using `clear`
    void backward();

    VariableRef operator + (VariableRef other);
    VariableRef operator - (VariableRef other);
    VariableRef operator * (VariableRef other);
    VariableRef operator / (VariableRef other);

    VariableRef matmul(VariableRef other);
    VariableRef sum();

    // Switch computations to cpu, but only on this node!
    void cpu();
    // Switch computations to gpu on this node
    void gpu();

    // Zero out the gradients recursively
    void clear();
};
