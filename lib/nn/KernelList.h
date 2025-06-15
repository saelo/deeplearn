//
// List of all available OpenCL kernels, their identifier and hosting program (== filename).
//
// Copyright (c) 2016 Samuel Groß
//
// This could be extended to auto-generate the C++ function stubs...
//

//Id,                                   Program,            Kernel
C(kMSEKernel,                           "Misc",             "MSE"),

C(kAddKernel,                           "Arithmetic",       "Add"),
C(kScaledAddKernel,                     "Arithmetic",       "ScaledAdd"),
C(kScalarAddKernel,                     "Arithmetic",       "ScalarAdd"),
C(kSubKernel,                           "Arithmetic",       "Sub"),
C(kScalarSubKernel,                     "Arithmetic",       "ScalarSub"),
C(kMulKernel,                           "Arithmetic",       "Mul"),
C(kScalarMulKernel,                     "Arithmetic",       "ScalarMul"),
C(kDivKernel,                           "Arithmetic",       "Div"),
C(kScalarDivKernel,                     "Arithmetic",       "ScalarDiv"),
C(kExpKernel,                           "Arithmetic",       "Exp"),
C(kLogKernel,                           "Arithmetic",       "Log"),

C(kSigmoidKernel,                       "Activations",      "Sigmoid"),
C(kSigmoidDerivativeKernel,             "Activations",      "SigmoidDerivative"),
C(kReLUKernel,                          "Activations",      "ReLU"),
C(kReLUDerivativeKernel,                "Activations",      "ReLUDerivative"),

C(kMatVecMulKernel,                     "LinearAlgebra",    "MatVecMul"),
C(kMatVecMulReduceKernel,               "LinearAlgebra",    "MatVecMulReduce"),
C(kTransposedMatVecMulKernel,           "LinearAlgebra",    "TransposedMatVecMul"),
C(kTransposedVecMulKernel,              "LinearAlgebra",    "TransposedVecMul"),

C(kMaxPool2DKernel,                     "Pooling",          "MaxPool2D"),
C(kMaxPool2DGradientsKernel,            "Pooling",          "MaxPool2DGradients"),
