//
// 2D Convolution layer.
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __CONVOLUTION_LAYER_H__
#define __CONVOLUTION_LAYER_H__

#include <cstddef>

#include "nn/Layer.h"
#include "nn/Tensor.h"
#include "common/Common.h"

namespace nn {

template <typename Tensor>
class ConvolutionLayer : public Layer<Tensor> {
  public:
    ConvolutionLayer(const Shape& input_shape, size_t num_features, size_t kernel_width, size_t kernel_height) :
        input_shape_(input_shape),
        output_shape_({num_features, input_shape[1], input_shape[2]}),
        // TODO GlorotInitializer doesn't seem to do well here... ?
        //kernels_({num_features, input_shape[0], kernel_height, kernel_width}, GlorotInitializer(input_shape[0] * kernel_width * kernel_height)),
        kernels_({num_features, input_shape[0], kernel_height, kernel_width}, RandomInitializer()),
        kernel_gradients_({num_features, input_shape[0], kernel_height, kernel_width}, ZeroInitializer),
        tmp_kernel_gradients_({num_features, input_shape[0], kernel_height, kernel_width}, ZeroInitializer),
        output_(output_shape_, ZeroInitializer),
        output_gradients_(input_shape_, ZeroInitializer),
        last_input_(nullptr) { }

    ConvolutionLayer(const Shape& input_shape, const Tensor& kernels) :
        input_shape_(input_shape),
        output_shape_({kernels.shape(0), input_shape[1], input_shape[2]}),
        kernels_(kernels),
        kernel_gradients_(kernels.shape(), ZeroInitializer),
        tmp_kernel_gradients_(kernels.shape(), ZeroInitializer),
        output_(output_shape_, ZeroInitializer),
        output_gradients_(input_shape_, ZeroInitializer),
        last_input_(nullptr)
    {
        Assert(kernels.rank() == 4);
        Assert(input_shape[0] == kernels.shape(1));
    }

    virtual ~ConvolutionLayer()
    {
    }

    virtual const Tensor& Forward(const Tensor& input) override
    {
        Assert(input.shape() == input_shape_);

        // We'll need our input later on during the backward pass.
        last_input_ = &input;

        convolution(input, kernels_, output_);

        return output_;
    }

    virtual const Tensor& Backward(const Tensor& gradients) override
    {
        Assert(gradients.shape() == output_shape_);

        // Calculate gradients for the kernel weights.
        // See the implementation for details. Basically this sums up
        // all the (input_pixel, output_pixel) pairs that each weight
        // of the kernel influenced.
        convolution_kernel_gradients(*last_input_, gradients, tmp_kernel_gradients_);

        // Sum of the kernel weight gradients for the current mini-batch.
        kernel_gradients_ += tmp_kernel_gradients_;

        // We get the derivatives of the loss function wrt to our inputs simply by
        // doing a cross convolution on it.
        //
        // Why? During the forward pass, image[i][j] influenced a number of surrounding
        // output values through a simple multiplication (which becomes a constant factor
        // when computing the derivative). We need to use the same kernel weight during the
        // backward pass, so we need to use a mirrored kernel ==> a cross-correlation.
        cross_correlation(gradients, kernels_, output_gradients_);

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
        add(kernels_, kernel_gradients_, -1 * (epsilon / batch_size), kernels_);
        kernel_gradients_.Clear();
    }

    virtual Tensor CurrentGradients() const override
    {
        return kernel_gradients_;
    }

  private:
    // 3D dimension of the input tensor: (channels, image_height, image_width).
    Shape input_shape_;

    // 3D dimension of the output tensor: (num_features, image_height, image_width).
    Shape output_shape_;

    // Convolution kernels. This is a tensor of shape (num_features, kernel_dim_y, kernel_dim_x);
    Tensor kernels_;

    // Gradients of the kernels during backpropagation.
    Tensor kernel_gradients_;

    // Hold the kernel gradients during one backward pass. Added up into kernel_gradients_ for a mini batch.
    Tensor tmp_kernel_gradients_;

    // Output tensor, populated during the forward pass.
    // This contains the output of this layer before the activation function is executed.
    Tensor output_;

    // Error output tensor, populated during the backward pass.
    Tensor output_gradients_;

    // Input during the forward pass, needed to calculate the gradients.
    // Pointer not owned by this instance.
    const Tensor* last_input_;


    DISALLOW_COPY_AND_ASSIGN(ConvolutionLayer);
};

}       // namespace nn

#endif
