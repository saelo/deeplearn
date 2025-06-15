//
// Softmax activation
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __SOFTMAX_ACTIVATION_H__
#define __SOFTMAX_ACTIVATION_H__

#include <cstddef>
#include <cmath>

#include "nn/Activation.h"
#include "common/Common.h"

namespace nn {

template <typename Tensor>
class SoftmaxActivation : public Activation<Tensor> {
  public:
    SoftmaxActivation(Shape shape) : output_(shape), last_input_(nullptr), shape_(shape) { }

    virtual const Tensor& Forward(const Tensor& input) override
    {
        Assert(input.shape() == shape_);

        last_input_ = &input;

        exp(input, output_);
        float s = sum(output_);
        output_ /= s;


        return output_;
    }

    virtual const Tensor& Backward(const Tensor& loss) override
    {
        Assert(loss.shape() == shape_);

        Check(false, "Softmax activation is currently only supported in combination with the cross-entropy objective");

        return output_;
    }

    virtual Shape InputTensorShape() const override { return shape_; }

    virtual const Tensor* Dispatch(Objective<Tensor>* objective, const Tensor& data) override
    {
        return objective->Accept(this, data);
    }

    const Tensor* last_output() const { return &output_; }

  private:
    // Output tensor for this activation.
    Tensor output_;

    // Input during the forward pass. Needed to calculate the gradients.
    const Tensor* last_input_;

    // Shape of the tensors processed by this activation instance.
    Shape shape_;

    DISALLOW_COPY_AND_ASSIGN(SoftmaxActivation);
};

}       // namespace nn

#endif
