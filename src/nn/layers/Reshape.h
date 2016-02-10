//
// Reshape "pseudo" layer.
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __RESHAPE_LAYER_H__
#define __RESHAPE_LAYER_H__

#include <cstddef>

#include "nn/Layer.h"
#include "common/Common.h"

namespace nn {

template <typename Tensor>
class ReshapeLayer : public Layer<Tensor> {
  public:
    ReshapeLayer(const Shape& input_shape, const Shape& output_shape) :
        input_shape_(input_shape),
        output_shape_(output_shape),
        output_(nullptr),
        output_gradients_(nullptr)
    {
        Assert(input_shape.TotalElementCount() == output_shape.TotalElementCount());
    }

    virtual ~ReshapeLayer()
    {
        if (output_)
            delete output_;
        if (output_gradients_)
            delete output_gradients_;
    }

    virtual const Tensor& Forward(const Tensor& input) override
    {
        Assert(input.shape() == input_shape_);

        if (output_)
            delete output_;
        output_ = input.NewView(output_shape_);
        return *output_;
    }

    virtual const Tensor& Backward(const Tensor& gradients) override
    {
        Assert(gradients.shape() == output_shape_);

        if (output_gradients_)
            delete output_gradients_;
        output_gradients_ = gradients.NewView(input_shape_);
        return *output_gradients_;
    }

    virtual Shape InputTensorShape() const override
    {
        return input_shape_;
    }

    virtual Shape OutputTensorShape() const override
    {
        return output_shape_;
    }

    virtual void GradientDescent(size_t batch_size, float epsilon) override
    {
        // nothing to do here
    }

  private:
    // Input and output tensor shape.
    Shape input_shape_;
    Shape output_shape_;

    // Pointer to a dependent tensor, created from the input tensor in the forward pass.
    const Tensor* output_;

    // Pointer to a dependent tensor, created from the input gradient tensor in the backward pass.
    const Tensor* output_gradients_;

    DISALLOW_COPY_AND_ASSIGN(ReshapeLayer);
};


}       // namespace nn

#endif
