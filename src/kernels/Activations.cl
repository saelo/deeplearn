#include "KernelCommon.h"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline float sigmoid(float f) {
    return 1.0 / (1.0 + exp(-f));
}

inline float sigmoid_derivative(float f) {
    return sigmoid(f) * (1.0 - sigmoid(f));
}

UNARY_OPERATION(Sigmoid, sigmoid);
UNARY_OPERATION(SigmoidDerivative, sigmoid_derivative);


inline float relu(float f) {
    return max(0.f, f);
}

inline float relu_derivative(float f) {
    return f < 0 ? 0 : 1;
}

UNARY_OPERATION(ReLU, relu);
UNARY_OPERATION(ReLUDerivative, relu_derivative);
