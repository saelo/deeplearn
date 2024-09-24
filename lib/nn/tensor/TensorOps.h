//
// Tensor operations
//
// Copyright (c) 2016 Samuel Groß
//

//
// !! DO NO INCLUDE THIS FILE DIRECTLY !!
//
// Instead, this file is include multiple times in the header files
// for the concrete tensor classes. There, the type 'Tensor' is #defined
// to the concrete tensor type.
//

//
// This file contains all operations that can be performed on any tensor. It
// is part of the public tensor API.
//
// Most operations (except the ones that yield a scalar value, like sum())
// take an output tensor as last argument which they also return.
// This is done to avoid memory allocations as it allows output tensors to be reused.
//

// Make autocompletion happy
#ifndef Tensor
#define Tensor float
#endif



//
// Matrix and vector operations
//
// Matrix-vector multiplication.
Tensor& matvecmul(const Tensor& matrix, const Tensor& vector, Tensor& output);

// Matrix-vector multiplication with transposed matrix.
Tensor& transposed_matvecmul(const Tensor& matrix, const Tensor& vector, Tensor& output);

// Vector-vector multiplication. Yields a scalar.
float vecmul(const Tensor& x, const Tensor& y);

// Transposed vector-vector multiplication. Yields a matrix of shape (x.shape(0), y.shape(0)).
Tensor& transposed_vecmul(const Tensor& x, const Tensor& y, Tensor& output);


//
// Pooling
//
// 2D Max pooling. Input shape: (num_channels, height, width).
Tensor& maxpool(const Tensor& input, size_t pooling_width, size_t pooling_height, Tensor& output);

// Gradient computation for a max pooling layer.
Tensor& maxpool_gradients(const Tensor& input, const Tensor& gradients, size_t pooling_width, size_t pooling_height, Tensor& output);


//
// Convolution
//
// 2D Convolution with zero padding at the borders.
//
// The input tensor is a tensor of shape (num_channels, height, width), the output
// is a tensor of shape (num_features, height, width).
// The kernel tensor must be a 4D tensor: (num_features, num_channels, kernel_height, kernel_width).
Tensor& convolution(const Tensor& input, const Tensor& kernels, Tensor& output);

// 2D Cross-correlation, a convolution without mirroring the kernel.
//
// The input tensor is a tensor of shape (num_features, height, width), the output
// is a tensor of shape (num_channels, height, width).
// The kernel tensor must be a 4D tensor: (num_features, num_channels, kernel_height, kernel_width).
//
// Note: This function is specifically modified (compared to convolution()) to support
// the ConvolutionLayer. There, the output of a previous convolution becomes the input
// to a cross-correlation, thus the change in the first index of input and output tensor.
Tensor& cross_correlation(const Tensor& input, const Tensor& kernels, Tensor& output);

// Gradient calculation for the weights of a 4D convolution kernel: (num_features, num_channels, kernel_height, kernel_width).
Tensor& convolution_kernel_gradients(const Tensor& input, const Tensor& gradients, Tensor& output);


//
// Elementwise operations
//
// output = x + v * y
Tensor& add(const Tensor& x, const Tensor& y, float v, Tensor& output);

// output = x + y
Tensor& add(const Tensor& x, const Tensor& y, Tensor& output);

// output = x - y
Tensor& sub(const Tensor& x, const Tensor& y, Tensor& output);

// output = x * y
Tensor& mul(const Tensor& x, const Tensor& y, Tensor& output);

// output = x / y
Tensor& div(const Tensor& x, const Tensor& y, Tensor& output);

// output = x + v
Tensor& add(const Tensor& x, float v, Tensor& output);

// output = x - v
Tensor& sub(const Tensor& x, float v, Tensor& output);

// output = x * v
Tensor& mul(const Tensor& x, float v, Tensor& output);

// output = x / v
Tensor& div(const Tensor& x, float v, Tensor& output);

// output = exp(input)
Tensor& exp(const Tensor& input, Tensor& output);

// output = log(input)
Tensor& log(const Tensor& input, Tensor& output);

// Sigmoid function
Tensor& sigmoid(const Tensor& input, Tensor& output);

// Derivative of the sigmoid function
Tensor& sigmoid_derivative(const Tensor& input, Tensor& output);

// ReLU function
Tensor& relu(const Tensor& input, Tensor& output);

// Derivative of the relu function
Tensor& relu_derivative(const Tensor& input, Tensor& output);


//
// Miscellaneous operations
//
// Returns the index of the largest element in the given tensor.
// The provided tensor should be one dimensional, otherwise a warning is printed.
size_t argmax(const Tensor& input);

// Returns the sum of all elements in the given tensor.
float sum(const Tensor& input);

// Returns the mean squared error between the given tensors.
float mse(const Tensor& x, const Tensor& y);
