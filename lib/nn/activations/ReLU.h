//
// ReLU activation
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __RELU_ACTIVATION_H__
#define __RELU_ACTIVATION_H__

#include <cstddef>
#include <cmath>

#include "nn/Activation.h"
#include "common/Common.h"

namespace nn {

template <typename Tensor>
class ReLUActivation : public Activation<Tensor> {
  public:
    ReLUActivation(Shape shape) : output_(shape), last_input_(nullptr), shape_(shape) { }

    virtual const Tensor& Forward(const Tensor& input) override
    {
        Assert(input.shape() == shape_);

        last_input_ = &input;
        relu(input, output_);

        return output_;
    }

    virtual const Tensor& Backward(const Tensor& gradients) override
    {
        Assert(gradients.shape() == shape_);

        relu_derivative(*last_input_, output_);
        output_ *= gradients;

        return output_;
    }

    virtual Shape InputTensorShape() const override { return shape_; }

    virtual const Tensor* Dispatch(Objective<Tensor>* objective, const Tensor& data) override
    {
        return objective->Accept(this, data);
    }

  private:
    // Output tensor for this activation.
    Tensor output_;

    // Input during the forward pass. Needed to calculate the gradients.
    const Tensor* last_input_;

    // Shape of the tensors processed by this activation instance.
    Shape shape_;

    DISALLOW_COPY_AND_ASSIGN(ReLUActivation);
};

}       // namespace nn

#endif
