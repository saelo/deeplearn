//
// Fully connected layer
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __DENSE_LAYER_H__
#define __DENSE_LAYER_H__

#include <cstddef>

#include "nn/Layer.h"
#include "nn/Tensor.h"
#include "common/Common.h"

namespace nn {

template <typename Tensor>
class DenseLayer : public Layer<Tensor> {
  public:
    DenseLayer(size_t input_dim, size_t output_dim) :
        weights_({output_dim, input_dim}, GlorotInitializer(input_dim)),
        output_({output_dim}),
        output_gradients_({input_dim}),
        tmp_weight_gradients_({output_dim, input_dim}, ZeroInitializer),
        weight_gradients_({output_dim, input_dim}, ZeroInitializer),
        last_input_(nullptr),
        input_dim_(input_dim),
        output_dim_(output_dim) { }

    DenseLayer(const Tensor& weights) :
        weights_(weights),
        output_({weights.shape(0)}),
        output_gradients_({weights.shape(1)}),
        tmp_weight_gradients_(weights.shape(), ZeroInitializer),
        weight_gradients_(weights.shape(), ZeroInitializer),
        last_input_(nullptr),
        input_dim_(weights.shape(1)),
        output_dim_(weights.shape(0)) { }

    virtual ~DenseLayer()
    {
    }

    virtual const Tensor& Forward(const Tensor& input) override
    {
        Assert(input.shape() == Shape({input_dim_}));

        // We'll need our input later on during the backward pass.
        last_input_ = &input;

        // Calculate weighted sum from every input neuron to every output neuron ==> matrix-vector multiplication.
        matvecmul(weights_, input, output_);

        return output_;
    }

    virtual const Tensor& Backward(const Tensor& gradients) override
    {
        Assert(gradients.shape() == Shape({output_dim_}));

        // Update weight derivatives.
        weight_gradients_ += transposed_vecmul(gradients, *last_input_, tmp_weight_gradients_);

        // "Reverse" the matrix-vector multiplication.
        transposed_matvecmul(weights_, gradients, output_gradients_);

        return output_gradients_;
    }

    virtual Shape InputTensorShape() const override
    {
        return Shape({input_dim_});
    }

    virtual Shape OutputTensorShape() const override
    {
        return Shape({output_dim_});
    }

    virtual void GradientDescent(size_t batch_size, float epsilon) override
    {
        add(weights_, weight_gradients_, -1 * (epsilon / batch_size), weights_);
        weight_gradients_.Clear();
    }

    virtual Tensor CurrentGradients() const override
    {
        return weight_gradients_;
    }

  private:
    // Weights and bias variables. These are learned during training.
    Tensor weights_;

    // Output tensor, populated during the forward pass.
    Tensor output_;

    // Error output tensor, populated during the backward pass.
    Tensor output_gradients_;

    // Tensors to sum up the partial derivatives for each minibatch during training.
    Tensor tmp_weight_gradients_;               // Used to store the outcome of the transposed vector-vector multiplication in.
    Tensor weight_gradients_;

    // Input during the forward pass, needed to calculate the gradients.
    // Pointer not owned by this instance.
    const Tensor* last_input_;

    // 1D dimension of the input tensor.
    size_t input_dim_;

    // 1D dimension of the output tensor.
    size_t output_dim_;


    DISALLOW_COPY_AND_ASSIGN(DenseLayer);
};

}       // namespace nn

#endif
