//
// Header file to include all public headers for nn/
//
// Copyright (c) 2016 Samuel Groß
//

// General stuff
#include "nn/Gpu.h"
#include "nn/Initializer.h"
#include "nn/Network.h"

// Tensors
#include "nn/Tensor.h"

// Layers
#include "nn/layers/Bias.h"
#include "nn/layers/Convolution.h"
#include "nn/layers/Dense.h"
#include "nn/layers/MaxPool.h"
#include "nn/layers/Reshape.h"

// Activations
#include "nn/activations/ReLU.h"
#include "nn/activations/Sigmoid.h"
#include "nn/activations/Softmax.h"

// Objectives
#include "nn/objectives/CrossEntropy.h"
#include "nn/objectives/MSE.h"

namespace nn {

// Namespaces to export all template classes for either CPU or GPU tensor instances.
namespace gpu {

using nn::GPUTensor;
using nn::CPUTensor;

typedef Network<GPUTensor> Network;

typedef ConvolutionLayer<GPUTensor> ConvolutionLayer;
typedef DenseLayer<GPUTensor> DenseLayer;
typedef MaxPool2DLayer<GPUTensor> MaxPool2DLayer;
typedef BiasLayer<GPUTensor> BiasLayer;
typedef ReshapeLayer<GPUTensor> ReshapeLayer;

typedef SigmoidActivation<GPUTensor> SigmoidActivation;
typedef ReLUActivation<GPUTensor> ReLUActivation;
typedef SoftmaxActivation<GPUTensor> SoftmaxActivation;

typedef MSE<GPUTensor> MSE;
typedef CrossEntropy<GPUTensor> CrossEntropy;

}       // namespace gpu

namespace cpu {

using nn::GPUTensor;
using nn::CPUTensor;

typedef Network<CPUTensor> Network;

typedef ConvolutionLayer<CPUTensor> ConvolutionLayer;
typedef DenseLayer<CPUTensor> DenseLayer;
typedef MaxPool2DLayer<CPUTensor> MaxPool2DLayer;
typedef BiasLayer<CPUTensor> BiasLayer;
typedef ReshapeLayer<CPUTensor> ReshapeLayer;

typedef SigmoidActivation<CPUTensor> SigmoidActivation;
typedef ReLUActivation<CPUTensor> ReLUActivation;
typedef SoftmaxActivation<CPUTensor> SoftmaxActivation;

typedef MSE<CPUTensor> MSE;
typedef CrossEntropy<CPUTensor> CrossEntropy;

}       // namespace gpu

}       // namespace nn
