//
// Base layer class
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __LAYER_H__
#define __LAYER_H__

#include "nn/Tensor.h"
#include "nn/Initializer.h"

namespace nn {

template <typename Tensor>
class Layer {
  public:
    // Default destructor.
    virtual ~Layer() { };

    // Forward pass of this layer.
    //
    // During the forward pass, each layer receives the output of the previous layer
    // (or the input tensor if it is the first layer of the network).
    virtual const Tensor& Forward(const Tensor& input) = 0;

    // Backward pass of this layer.
    //
    // Each layer receives the gradients of the loss function with regard to
    // the layers outputs during the last forward pass.
    // From there the layer is required to calculate the gradients of the
    // loss function wrt its inputs as well as to any variables that it stores (weights, biases).
    //
    // This function returns the gradients of the loss function wrt the output of the previous layer.
    virtual const Tensor& Backward(const Tensor& gradients) = 0;

    // Returns the input tensor shape of this layer.
    virtual Shape InputTensorShape() const = 0;

    // Returns the output tensor shape of this layer.
    virtual Shape OutputTensorShape() const = 0;

    // Perform a gradient descent step on the previously processed mini batch.
    virtual void GradientDescent(size_t batch_size, float epsilon) = 0;

    // Returns a tensor holding the current weight gradients.
    // This is mostly useful for testing purposes.
    virtual Tensor CurrentGradients() const { return Tensor(); }
};

}       // namespace nn

#endif
