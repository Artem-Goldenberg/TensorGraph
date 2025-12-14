#pragma once

#include <cassert>
#include "device-tensor.h"
#include "primitives.h"

namespace gpu {

class Tensor final : public DeviceTensor {
public:
    Tensor(const TensorParams& params, const void* data);

    template <typename Data>
    Tensor(Data elem, const TensorParams& params);

    TensorRef copy() const override;

    void add(const DeviceTensor& other) override;
    void subtract(const DeviceTensor& other) override;
    void multiply(const DeviceTensor& other) override;
    void divide(const DeviceTensor& other) override;

    TensorRef sum() const override;

    TensorRef matmul(const DeviceTensor& other) const override;
    TensorRef transpose() const override;

    void clear() override;

    template <typename Data>
    void flush(Data* out) const;

    int get_grid_size() const { return grid_size; }
    int get_block_size() const { return block_size; }

private:
    void compute_grid_sizes();

    int grid_size;
    int block_size;
};

}
