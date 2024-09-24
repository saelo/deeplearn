//
// Bias layer, simply adds each weight to the input tensor.
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __BIAS_LAYER_H__
#define __BIAS_LAYER_H__

#include <cstddef>

#include "nn/Layer.h"
#include "nn/Tensor.h"
#include "common/Common.h"

namespace nn {

template <typename Tensor>
class BiasLayer : public Layer<Tensor> {
  public:
    BiasLayer(const Shape& shape) :
        weights_(shape, RandomInitializer()),
        output_(shape),
        gradients_(shape, ZeroInitializer),
        last_input_(nullptr),
        shape_(shape) { }

    BiasLayer(const Tensor& weights) :
        weights_(weights),
        output_(weights.shape()),
        gradients_(weights.shape()),
        last_input_(nullptr),
        shape_(weights.shape()) { }

    virtual ~BiasLayer()
    {
    }

    virtual const Tensor& Forward(const Tensor& input) override
    {
        return add(input, weights_, output_);
    }

    virtual const Tensor& Backward(const Tensor& gradients) override
    {
        // Weight gradients are equal to the output gradients since $dO/dx = d/dx x + y = 1$ ==> $dL/dx = dL/do * 1$
        gradients_ += gradients;

        // Same for the input gradients.
        return gradients;
    }

    virtual Shape InputTensorShape() const override
    {
        return shape_;
    }

    virtual Shape OutputTensorShape() const override
    {
        return shape_;
    }

    virtual void GradientDescent(size_t batch_size, float epsilon) override
    {
        add(weights_, gradients_, -1 * (epsilon / batch_size), weights_);
        gradients_.Clear();
    }

  private:
    // Learnable weights of this layer.
    Tensor weights_;

    // Output tensor, populated during the forward pass.
    Tensor output_;

    // Tensors to sum up the partial derivatives for each minibatch during training.
    Tensor gradients_;

    // Input during the forward pass, needed to calculate the gradients.
    // Pointer not owned by this instance.
    const Tensor* last_input_;

    // Shape of input and output tensor.
    Shape shape_;


    DISALLOW_COPY_AND_ASSIGN(BiasLayer);
};

}       // namespace nn

#endif
