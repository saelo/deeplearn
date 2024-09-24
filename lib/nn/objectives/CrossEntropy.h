//
// Optimizer to minimize the cross entropy.
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __CROSS_ENTROPY_OBJECTIVE_H__
#define __CROSS_ENTROPY_OBJECTIVE_H__

#include "nn/Objective.h"

namespace nn {

template <typename Tensor>
class CrossEntropy : public Objective<Tensor> {
  public:
    CrossEntropy(const Shape& network_output_shape) : gradients_(network_output_shape), network_output_logarithms_(network_output_shape)
    {
        // For now we only support vectors as the output of our networks.
        Assert(network_output_shape.rank() == 1);
    }

    virtual float Loss(const Tensor& network_output, const Tensor& label) override
    {
        Assert(network_output.shape() == label.shape());
        Assert(network_output.shape() == gradients_.shape());

        log(network_output, network_output_logarithms_);
        mul(network_output_logarithms_, label, network_output_logarithms_);
        return -sum(network_output_logarithms_);
    }

    virtual const Tensor& LossGradientWrtNetworkOutput(const Tensor& network_output, const Tensor& label) override
    {
        // For now cross-entropy is only supported if the last layer is a Softmax activation, in which
        // case LossGradientWrtActivationInput will calculate the correct gradients.
        Check(false, "Cross-entropy is only supported in combination with a Softmax as final layer.");
        return gradients_;
    }

    virtual const Tensor* Accept(SoftmaxActivation<Tensor>* softmax, const Tensor& label) override
    {
        Assert(softmax->last_output()->shape() == label.shape());
        Assert(softmax->last_output()->shape() == gradients_.shape());

        // See Math.md for an explanation.
        sub(*softmax->last_output(), label, gradients_);
        return &gradients_;
    }

  private:
    // Storage for the gradients to avoid memory allocations.
    Tensor gradients_;

    // Storage for the network output logarithms, needed during the loss calculation.
    Tensor network_output_logarithms_;
};

}       // namespace nn

#endif
