//
// Optimizer to minimize the mean squared error (MSE).
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __MSE_OBJECTIVE_H__
#define __MSE_OBJECTIVE_H__

#include "nn/Objective.h"

namespace nn {

// Mean-squared error: C = 0.5 * sum((y - a)^2)
//
// We multiply by 0.5 so that the gradients become simpler.
template <typename Tensor>
class MSE : public Objective<Tensor> {
  public:
    MSE(const Shape& network_output_shape) : gradients_(network_output_shape)
    {
        // For now we only support vectors as the output of our networks.
        Assert(network_output_shape.rank() == 1);
    }

    virtual float Loss(const Tensor& network_output, const Tensor& label) override
    {
        Assert(network_output.shape() == label.shape());
        Assert(network_output.shape().rank() == 1);

        return 0.5 * mse(network_output, label);
    }

    virtual const Tensor& LossGradientWrtNetworkOutput(const Tensor& network_output, const Tensor& label) override
    {
        // dL/da_j = d/da_j (0.5 * (y - da_j)^2)
        //         = (-1) * (y - da_j)
        //         = da_j - y
        Assert(network_output.shape() == label.shape());

        sub(network_output, label, gradients_);
        return gradients_;
    }

  private:
    Tensor gradients_;
};

}       // namespace nn

#endif
