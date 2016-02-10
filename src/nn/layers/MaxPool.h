//
// 2D Max pooling layer.
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __MAXPOOL_LAYER_H__
#define __MAXPOOL_LAYER_H__

#include <cstddef>

#include "nn/Layer.h"
#include "nn/Tensor.h"
#include "common/Common.h"

namespace nn {

template <typename Tensor>
class MaxPool2DLayer : public Layer<Tensor> {
  public:
    MaxPool2DLayer(const Shape& input_shape, size_t x, size_t y) :
        input_shape_(input_shape),
        output_shape_({input_shape[0], (input_shape[1] + y - 1) / y, (input_shape[2] + x - 1) / x}),
        pooling_size_x_(x),
        pooling_size_y_(y),
        output_(output_shape_),
        output_gradients_(input_shape_),
        last_input_(nullptr)
    {
        // It's already too late here...
        Assert(input_shape.rank() == 3);
    }

    virtual ~MaxPool2DLayer()
    {
    }

    virtual const Tensor& Forward(const Tensor& input) override
    {
        Assert(input.shape() == input_shape_);

        // We'll need our input later on during the backward pass.
        last_input_ = &input;

        // Do the max pooling.
        maxpool(input, pooling_size_x_, pooling_size_y_, output_);

        return output_;
    }

    virtual const Tensor& Backward(const Tensor& gradients) override
    {
        Assert(gradients.shape() == output_shape_);

        // Undo the max pooling.
        maxpool_gradients(*last_input_, gradients, pooling_size_x_, pooling_size_y_, output_gradients_);

        return output_gradients_;
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
        // Nothing to do here.
    }

  private:
    // 3D dimension of the input tensor.
    Shape input_shape_;

    // 3D dimension of the output tensor.
    Shape output_shape_;

    // Pooling sizes.
    size_t pooling_size_x_, pooling_size_y_;

    // Output tensor, populated during the forward pass.
    // This contains the output of this layer before the activation function is executed.
    Tensor output_;

    // Error output tensor, populated during the backward pass.
    Tensor output_gradients_;

    // Input during the forward pass, needed to calculate the gradients.
    // Pointer not owned by this instance.
    const Tensor* last_input_;


    DISALLOW_COPY_AND_ASSIGN(MaxPool2DLayer);
};

}       // namespace nn

#endif
