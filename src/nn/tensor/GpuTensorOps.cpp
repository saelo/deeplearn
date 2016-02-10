#include <cfloat>
#include <memory>
#include <cmath>

#include "nn/tensor/GpuTensor.h"
#include "nn/tensor/CpuTensor.h"
#include "nn/Gpu.h"

#define INCLUDED_BY_HOST
#include "kernels/KernelCommon.h"
#undef INCLUDED_BY_HOST

// Some of the simple kernels can process multiple items per thread.
constexpr size_t kItemsPerThread = ITEMS_PER_THREAD;

// Number of threads to spawn if each thread processes |kItemsPerThread| items.
inline size_t threadcount(size_t problem_size)
{
    return (problem_size + kItemsPerThread - 1) / kItemsPerThread;
}

using namespace std;

namespace nn {

typedef ocl::Kernel::WorkSize WorkSize;

size_t argmax(const GPUTensor& input)
{
    return argmax(input.ToHost());
}

float sum(const GPUTensor& input)
{
    // Right now it's actually faster to sum up everything on the CPU since we need
    // a data transfer anyways and our vector sizes are always very small (10-100 elements).
    return sum(input.ToHost());
}

float mse(const GPUTensor& x, const GPUTensor& y)
{
    Assert(x.shape() == y.shape());

    GPUTensor errors(x.shape());
    bool success = GPUContext::kernel_manager.kernel(kMSEKernel)->Run(
            WorkSize(x.size()),
            x.size(),
            x.gpu_buffer(),
            y.gpu_buffer(),
            errors.gpu_buffer());
    Assert(success);

    return sum(errors);
}

GPUTensor& matvecmul(const GPUTensor& matrix, const GPUTensor& vector, GPUTensor& output)
{
    Assert(matrix.rank() == 2 && vector.rank() == 1 && output.rank() == 1);
    Assert(matrix.shape(0) == output.shape(0));
    Assert(matrix.shape(1) == vector.shape(0));

    // Each thread processes this many elements.
    size_t num_elements_per_thread = min((size_t)64, matrix.shape(1));

    // This many values will be produced for each row.
    size_t entries_per_row = (matrix.shape(COL) + num_elements_per_thread - 1) / num_elements_per_thread;

    GPUTensor temp_out({matrix.shape(ROW), entries_per_row});

    bool success = GPUContext::kernel_manager.kernel(kMatVecMulKernel)->Run(
            WorkSize(entries_per_row, matrix.shape(ROW)),
            WorkSize(1, 256),               // Required by kernel
            matrix.shape(ROW),
            matrix.shape(COL),
            num_elements_per_thread,
            matrix.gpu_buffer(),
            vector.gpu_buffer(),
            ocl::LocalMemory(num_elements_per_thread * sizeof(float)),
            temp_out.gpu_buffer());
    Assert(success);

    success = GPUContext::kernel_manager.kernel(kMatVecMulReduceKernel)->Run(
            WorkSize(output.shape(0)),
            output.shape(0),
            entries_per_row,
            temp_out.gpu_buffer(),
            output.gpu_buffer());
    Assert(success);

    return output;
}

GPUTensor& transposed_matvecmul(const GPUTensor& matrix, const GPUTensor& vector, GPUTensor& output)
{
    Assert(matrix.rank() == 2 && vector.rank() == 1 && output.rank() == 1);
    Assert(matrix.shape(0) == vector.shape(0));
    Assert(matrix.shape(1) == output.shape(0));

    // Each thread processes this many elements.
    size_t num_elements_per_thread = min((size_t)64, matrix.shape(1));

    // This many values will be produced for each row.
    size_t entries_per_row = (matrix.shape(ROW) + num_elements_per_thread - 1) / num_elements_per_thread;

    GPUTensor temp_out({matrix.shape(COL), entries_per_row});

    bool success = GPUContext::kernel_manager.kernel(kTransposedMatVecMulKernel)->Run(
            WorkSize(matrix.shape(COL), entries_per_row),
            WorkSize(256, 1),
            matrix.shape(ROW),
            matrix.shape(COL),
            num_elements_per_thread,
            matrix.gpu_buffer(),
            vector.gpu_buffer(),
            ocl::LocalMemory(num_elements_per_thread * sizeof(float)),
            temp_out.gpu_buffer());
    Assert(success);

    success = GPUContext::kernel_manager.kernel(kMatVecMulReduceKernel)->Run(
            WorkSize(output.shape(0)),
            output.shape(0),
            entries_per_row,
            temp_out.gpu_buffer(),
            output.gpu_buffer());
    Assert(success);

    return output;
}

float vecmul(const GPUTensor& x, const GPUTensor& y)
{
    Assert(x.rank() == 1);
    Assert(x.shape() == y.shape());

    GPUTensor tmp(x.shape());
    mul(x, y, tmp);

    return sum(tmp);
}

GPUTensor& transposed_vecmul(const GPUTensor& x, const GPUTensor& y, GPUTensor& output)
{
    Assert(x.rank() == 1 && y.rank() == 1 && output.rank() == 2);
    Assert(output.shape(0) == x.shape(0));
    Assert(output.shape(1) == y.shape(0));

    bool success = GPUContext::kernel_manager.kernel(kTransposedVecMulKernel)->Run(
            WorkSize(output.shape(0), output.shape(1)),
            output.shape(0),
            output.shape(1),
            x.gpu_buffer(),
            y.gpu_buffer(),
            output.gpu_buffer());
    Assert(success);

    return output;
}

#define UNARY_OPERATION(name, kernel_name) GPUTensor& name(const GPUTensor& input, GPUTensor& output)   \
{                                                                                                       \
    Assert(input.shape() == output.shape());                                                            \
                                                                                                        \
    bool success = GPUContext::kernel_manager.kernel(kernel_name)->Run(                                 \
            WorkSize(threadcount(input.size())),                                                        \
            input.size(),                                                                               \
            input.gpu_buffer(),                                                                         \
            output.gpu_buffer());                                                                       \
    Assert(success);                                                                                    \
                                                                                                        \
    return output;                                                                                      \
}

#define BINARY_OPERATION(name, kernel_name) GPUTensor& name(const GPUTensor& x, const GPUTensor& y,     \
        GPUTensor& output)                                                                              \
{                                                                                                       \
    Assert(x.shape() == y.shape());                                                                     \
    Assert(y.shape() == output.shape());                                                                \
                                                                                                        \
    bool success = GPUContext::kernel_manager.kernel(kernel_name)->Run(                                 \
            WorkSize(threadcount(x.size())),                                                            \
            x.size(),                                                                                   \
            x.gpu_buffer(),                                                                             \
            y.gpu_buffer(),                                                                             \
            output.gpu_buffer());                                                                       \
    Assert(success);                                                                                    \
                                                                                                        \
    return output;                                                                                      \
}

#define TENSOR_SCALAR_OPERATION(name, kernel_name) GPUTensor& name(const GPUTensor& x,                  \
        float v, GPUTensor& output)                                                                     \
{                                                                                                       \
    Assert(x.shape() == output.shape());                                                                \
                                                                                                        \
    bool success = GPUContext::kernel_manager.kernel(kernel_name)->Run(                                 \
            WorkSize(threadcount(x.size())),                                                            \
            x.size(),                                                                                   \
            x.gpu_buffer(),                                                                             \
            v,                                                                                          \
            output.gpu_buffer());                                                                       \
    Assert(success);                                                                                    \
                                                                                                        \
    return output;                                                                                      \
}

GPUTensor& add(const GPUTensor& x, const GPUTensor& y, float v, GPUTensor& output)
{
    Assert(x.shape() == y.shape());
    Assert(y.shape() == output.shape());

    bool success = GPUContext::kernel_manager.kernel(kScaledAddKernel)->Run(
            WorkSize(threadcount(x.size())),
            x.size(),
            x.gpu_buffer(),
            y.gpu_buffer(),
            v,
            output.gpu_buffer());
    Assert(success);

    return output;
}

BINARY_OPERATION(add, kAddKernel);
TENSOR_SCALAR_OPERATION(add, kScalarAddKernel);

BINARY_OPERATION(sub, kSubKernel);
TENSOR_SCALAR_OPERATION(sub, kScalarSubKernel);

BINARY_OPERATION(mul, kMulKernel);
TENSOR_SCALAR_OPERATION(mul, kScalarMulKernel);

BINARY_OPERATION(div, kDivKernel);
TENSOR_SCALAR_OPERATION(div, kScalarDivKernel);

UNARY_OPERATION(exp, kExpKernel);
UNARY_OPERATION(log, kLogKernel);

UNARY_OPERATION(sigmoid, kSigmoidKernel);
UNARY_OPERATION(sigmoid_derivative, kSigmoidDerivativeKernel);
UNARY_OPERATION(relu, kReLUKernel);
UNARY_OPERATION(relu_derivative, kReLUDerivativeKernel);


GPUTensor& maxpool(const GPUTensor& input, size_t pooling_width, size_t pooling_height, GPUTensor& output)
{
    Assert(input.rank() == 3 && output.rank() == 3);
    Assert(input.shape(0) == output.shape(0));
    Assert((input.shape(1) + pooling_height - 1) / pooling_height == output.shape(1));
    Assert((input.shape(2) + pooling_width - 1) / pooling_width == output.shape(2));

    // One thread per element input the output tensor.
    bool success = GPUContext::kernel_manager.kernel(kMaxPool2DKernel)->Run(
            WorkSize(output.shape(2), output.shape(1), output.shape(0)),
            output.shape(2),
            output.shape(1),
            output.shape(0),
            input.shape(2),
            input.shape(1),
            pooling_width,
            pooling_height,
            input.gpu_buffer(),
            output.gpu_buffer());
    Assert(success);

    return output;
}

GPUTensor& maxpool_gradients(const GPUTensor& input, const GPUTensor& gradients, size_t pooling_width, size_t pooling_height, GPUTensor& output)
{
    Assert(input.rank() == 3 && output.rank() == 3);
    Assert(input.shape(0) == output.shape(0));
    Assert((input.shape(1) + pooling_height - 1) / pooling_height == gradients.shape(1));
    Assert((input.shape(2) + pooling_width - 1) / pooling_width == gradients.shape(2));

    output.Clear();
    bool success = GPUContext::kernel_manager.kernel(kMaxPool2DGradientsKernel)->Run(
            WorkSize(output.shape(2), output.shape(1), output.shape(0)),
            gradients.shape(2),
            gradients.shape(1),
            gradients.shape(0),
            input.shape(2),
            input.shape(1),
            pooling_width,
            pooling_height,
            input.gpu_buffer(),
            gradients.gpu_buffer(),
            output.gpu_buffer());
    Assert(success);

    return output;
}

GPUTensor& convolution(const GPUTensor& input, const GPUTensor& kernels, GPUTensor& output)
{
    Assert(kernels.rank() == 4);
    Assert(input.rank() == 3 && output.rank() == 3);
    Assert(kernels.shape(2) % 2 == 1 && kernels.shape(3) % 2 == 1);
    Assert(kernels.shape(0) == output.shape(0) && kernels.shape(1) == input.shape(0));
    Assert(input.shape().ElementShape() == output.shape().ElementShape());
    Assert(kernels.shape(2) < kMaxConvolutionKernelSize && kernels.shape(3) < kMaxConvolutionKernelSize);

    for (size_t channel = 0; channel < input.shape(0); channel++) {
        bool success = GPUContext::kernel_manager.convolution_kernel(kernels.shape(3), kernels.shape(2))->Run(
                WorkSize(output.shape(2), output.shape(1), output.shape(0)),
                WorkSize(16, 16, 1),           // Kernel requires specific work group size
                output.shape(2),
                output.shape(1),
                channel,
                input.shape(0),
                input.gpu_buffer(),
                kernels.gpu_buffer(),
                output.gpu_buffer());
        Assert(success);
    }

    return output;
}

GPUTensor& cross_correlation(const GPUTensor& input, const GPUTensor& kernels, GPUTensor& output)
{
    Assert(kernels.rank() == 4);
    Assert(input.rank() == 3 && output.rank() == 3);
    Assert(kernels.shape(2) % 2 == 1 && kernels.shape(3) % 2 == 1);
    Assert(kernels.shape(0) == input.shape(0) && kernels.shape(1) == output.shape(0));
    Assert(input.shape().ElementShape() == output.shape().ElementShape());
    Assert(kernels.shape(2) < kMaxConvolutionKernelSize && kernels.shape(3) < kMaxConvolutionKernelSize);

    for (size_t channel = 0; channel < input.shape(0); channel++) {
        bool success = GPUContext::kernel_manager.cross_correlation_kernel(kernels.shape(3), kernels.shape(2))->Run(
                WorkSize(output.shape(2), output.shape(1), output.shape(0)),
                WorkSize(16, 16, 1),
                output.shape(2),
                output.shape(1),
                channel,
                output.shape(0),
                input.gpu_buffer(),
                kernels.gpu_buffer(),
                output.gpu_buffer());
        Assert(success);
    }

    return output;
}

GPUTensor& convolution_kernel_gradients(const GPUTensor& input, const GPUTensor& gradients, GPUTensor& kernels)
{
    Assert(kernels.rank() == 4);
    Assert(input.rank() == 3 && gradients.rank() == 3);
    Assert(kernels.shape(2) % 2 == 1 && kernels.shape(3) % 2 == 1);
    Assert(kernels.shape(0) == gradients.shape(0) && kernels.shape(1) == input.shape(0));
    Assert(input.shape().ElementShape() == gradients.shape().ElementShape());
    Assert(kernels.shape(2) < kMaxConvolutionKernelSize && kernels.shape(3) < kMaxConvolutionKernelSize);

    size_t kernel_size = kernels.shape(2) * kernels.shape(3);

    bool success = GPUContext::kernel_manager.convolution_gradient_kernel(kernels.shape(3), kernels.shape(2))->Run(
            WorkSize(kernel_size, input.shape(0), gradients.shape(0)),
            WorkSize(kernel_size, 1, 1),
            input.shape(2),
            input.shape(1),
            input.shape(0),
            input.gpu_buffer(),
            gradients.gpu_buffer(),
            kernels.gpu_buffer());
    Assert(success);

    return kernels;
}

}       // namespace nn
