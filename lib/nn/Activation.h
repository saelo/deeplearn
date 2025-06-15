//
// Base activation class
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include "nn/Tensor.h"
#include "nn/Layer.h"
#include "nn/Objective.h"

namespace nn {

// Base class for activations.
//
// An activation can be used as a standalone layer, but also as argument to another layer.
template <typename Tensor>
class Activation : public Layer<Tensor> {
  public:
    // Default destructor.
    virtual ~Activation() { };

    virtual const Tensor& Forward(const Tensor& input) = 0;
    virtual const Tensor& Backward(const Tensor& input) = 0;

    // Since activations never change the shape of the input tensors,
    // input and output tensor shapes are equal.
    virtual Shape InputTensorShape() const = 0;
    virtual Shape OutputTensorShape() const { return InputTensorShape(); }

    // Activations don't have weights or biases.
    virtual void GradientDescent(size_t batch_size, float epsilon) { }

    // See the comment in Objective.h for LossGradientWrtActivationInput.
    virtual const Tensor* Dispatch(Objective<Tensor>* objective, const Tensor& data) = 0;
};

}       // namespace nn

#endif
