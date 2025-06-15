//
// Base optimizer class
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __OBJECTIVE_H__
#define __OBJECTIVE_H__

#include "nn/Tensor.h"

namespace nn {

// Forward declarations for the different activations.
// This file is in turn included in Activation.h, so need to forward declare these here...
template <typename Tensor>
class Activation;
template <typename Tensor>
class ReLUActivation;
template <typename Tensor>
class SigmoidActivation;
template <typename Tensor>
class SoftmaxActivation;

// Base class for objectives.
//
// The objective is the value that should optimized by the network.
template <typename Tensor>
class Objective {
  public:
    // Default destructor.
    virtual ~Objective() { };

    // Calculate the loss.
    virtual float Loss(const Tensor& network_output, const Tensor& label) = 0;

    // Calculate the gradient of the loss function with regard to the output of the network.
    virtual const Tensor& LossGradientWrtNetworkOutput(const Tensor& network_output, const Tensor& label) = 0;

    // If the last layer is an activation (or more generally if the last layer does not have any trainable weights)
    // then it is possible and might be desirable to calculate the gradient of the loss wrt the input of the final layer
    // as opposed to the output of the final layer.
    //
    // This mechanism triggers a double dispatch on the provided activation object so that each Objective can
    // implement the corresponding methods only for the activations that it knows how to deal with.
    //
    // This method either return a valid pointer to a tensor which contains the gradients of the loss function
    // wrt to the input of the final activation, or nullptr if the gradient calculation for that activation not available.
    const Tensor* LossGradientWrtActivationInput(Activation<Tensor>* activation, const Tensor& label)
    {
        return activation->Dispatch(this, label);
    }

    // One method per Activation class. These perform the actual math for LossGradientWrtActivationInput.
    virtual const Tensor* Accept(ReLUActivation<Tensor>* relu, const Tensor& label) { return nullptr; }
    virtual const Tensor* Accept(SigmoidActivation<Tensor>* sigmoid, const Tensor& label) { return nullptr; }
    virtual const Tensor* Accept(SoftmaxActivation<Tensor>* softmax, const Tensor& label) { return nullptr; }
};

}       // namespace nn

#endif
