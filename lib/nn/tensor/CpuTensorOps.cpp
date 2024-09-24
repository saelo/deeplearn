#include <cfloat>
#include <memory>
#include <cmath>

#include "nn/tensor/CpuTensor.h"

namespace nn {

size_t argmax(const CPUTensor& input)
{
    Assert(input.rank() > 0);
    WARN_IF(input.rank() != 1, "argmax() called on tensor with rank > 1");

    float cmax = FLT_MIN;
    uint32_t imax = 0, i = 0;

    for (float v : input) {
        if (v > cmax)
            cmax = v, imax = i;
        i++;
    }

    return imax;
}

float mse(const CPUTensor& x, const CPUTensor& y)
{
    Assert(x.shape() == y.shape());

    float err = 0;
    for (size_t i = 0; i < x.shape(0); i++) {
        err += std::pow(y(i) - x(i), 2);
    }

    return err;
}

float sum(const CPUTensor& input)
{
    float sum = 0.f;
    for (float v : input)
        sum += v;

    return sum;
}

CPUTensor& matvecmul(const CPUTensor& matrix, const CPUTensor& vector, CPUTensor& output)
{
    Assert(matrix.rank() == 2 && vector.rank() == 1 && output.rank() == 1);
    Assert(matrix.shape(0) == output.shape(0));
    Assert(matrix.shape(1) == vector.shape(0));

    for (size_t row = 0; row < matrix.shape(0); row++) {
        output(row) = 0;
        for (size_t col = 0; col < matrix.shape(1); col++) {
            output(row) += matrix(row, col) * vector(col);
        }
    }

    return output;
}

CPUTensor& transposed_matvecmul(const CPUTensor& matrix, const CPUTensor& vector, CPUTensor& output)
{
    Assert(matrix.rank() == 2 && vector.rank() == 1 && output.rank() == 1);
    Assert(matrix.shape(0) == vector.shape(0));
    Assert(matrix.shape(1) == output.shape(0));

    for (size_t col = 0; col < matrix.shape(1); col++) {
        output(col) = 0;
        for (size_t row = 0; row < matrix.shape(0); row++) {
            output(col) += matrix(row, col) * vector(row);
        }
    }

    return output;
}

float vecmul(const CPUTensor& x, const CPUTensor& y)
{
    Assert(x.rank() == 1);
    Assert(x.shape() == y.shape());

    float res = 0;

    for (size_t i = 0; i < x.shape(0); i++)
        res += x(i) * y(i);

    return res;
}

CPUTensor& transposed_vecmul(const CPUTensor& x, const CPUTensor& y, CPUTensor& output)
{
    Assert(x.rank() == 1 && y.rank() == 1 && output.rank() == 2);
    Assert(output.shape(0) == x.shape(0));
    Assert(output.shape(1) == y.shape(0));

    for (size_t row = 0; row < x.shape(0); row++) {
        for (size_t col = 0; col < y.shape(0); col++) {
            output(row, col) = x(row) * y(col);
        }
    }

    return output;
}

#define UNARY_OPERATION(name, op) CPUTensor& name(const CPUTensor& input, CPUTensor& output)                        \
{                                                                                                                   \
    Assert(input.shape() == output.shape());                                                                        \
                                                                                                                    \
    auto i = input.begin();                                                                                         \
    auto o = output.begin();                                                                                        \
    for (; i != input.end(); i++, o++)                                                                              \
        *o = op(*i);                                                                                                \
                                                                                                                    \
    return output;                                                                                                  \
}

#define BINARY_OPERATION(name, op) CPUTensor& name(const CPUTensor& x, const CPUTensor& y, CPUTensor& output)       \
{                                                                                                                   \
    Assert(x.shape() == y.shape());                                                                                 \
    Assert(y.shape() == output.shape());                                                                            \
                                                                                                                    \
    auto i = x.begin(), j = y.begin();                                                                              \
    auto o = output.begin();                                                                                        \
    for (; i != x.end(); i++, j++, o++)                                                                             \
        *o = op((*i), (*j));                                                                                        \
                                                                                                                    \
    return output;                                                                                                  \
}

#define TENSOR_SCALAR_OPERATION(name, op) CPUTensor& name(const CPUTensor& x, float v, CPUTensor& output)           \
{                                                                                                                   \
    Assert(x.shape() == output.shape());                                                                            \
                                                                                                                    \
    auto i = x.begin();                                                                                             \
    auto o = output.begin();                                                                                        \
    for (; i != x.end(); i++, o++)                                                                                  \
        *o = op((*i), v);                                                                                           \
                                                                                                                    \
    return output;                                                                                                  \
}

CPUTensor& add(const CPUTensor& x, const CPUTensor& y, float f, CPUTensor& output)
{
    Assert(x.shape() == y.shape());
    Assert(y.shape() == output.shape());

    auto i = x.begin(), j = y.begin();
    auto o = output.begin();
    for (; i != x.end(); i++, j++, o++)
        *o = (*i) + (*j) * f;

    return output;
}

static inline float scalar_add(float x, float y) { return x + y; }
BINARY_OPERATION(add, scalar_add);
TENSOR_SCALAR_OPERATION(add, scalar_add);

static inline float scalar_sub(float x, float y) { return x - y; }
BINARY_OPERATION(sub, scalar_sub);
TENSOR_SCALAR_OPERATION(sub, scalar_sub);

static inline float scalar_mul(float x, float y) { return x * y; }
BINARY_OPERATION(mul, scalar_mul);
TENSOR_SCALAR_OPERATION(mul, scalar_mul);

static inline float scalar_div(float x, float y) { return x / y; }
BINARY_OPERATION(div, scalar_div);
TENSOR_SCALAR_OPERATION(div, scalar_div);

UNARY_OPERATION(exp, std::exp);
UNARY_OPERATION(log, std::log);


CPUTensor& maxpool(const CPUTensor& input, size_t pooling_width, size_t pooling_height, CPUTensor& output)
{
    Assert(input.rank() == 3 && output.rank() == 3);
    Assert(input.shape(0) == output.shape(0));
    Assert((input.shape(1) + pooling_height - 1) / pooling_height == output.shape(1));
    Assert((input.shape(2) + pooling_width - 1) / pooling_width == output.shape(2));

    for (size_t channel = 0; channel < input.shape(0); channel++) {
        for (size_t y = 0; y < input.shape(1); y += pooling_height) {
            for (size_t x = 0; x < input.shape(2); x += pooling_width) {
                float curmax = FLT_MIN;
                for (size_t oy = 0; oy < pooling_height; oy++) {
                    for (size_t ox = 0; ox < pooling_width; ox++) {
                        if (y + oy < input.shape(1) && x + ox < input.shape(2))
                            curmax = std::max(curmax, input(channel, y + oy, x + ox));
                    }
                }
                output(channel, y / pooling_height, x / pooling_width) = curmax;
            }
        }
    }

    return output;
}

CPUTensor& maxpool_gradients(const CPUTensor& input, const CPUTensor& gradients, size_t pooling_width, size_t pooling_height, CPUTensor& output)
{
    Assert(input.rank() == 3 && output.rank() == 3);
    Assert(input.shape(0) == output.shape(0));
    Assert((input.shape(1) + pooling_height - 1) / pooling_height == gradients.shape(1));
    Assert((input.shape(2) + pooling_width - 1) / pooling_width == gradients.shape(2));

    output.Clear();

    for (size_t channel = 0; channel < input.shape(0); channel++) {
        for (size_t y = 0; y < input.shape(1); y += pooling_height) {
            for (size_t x = 0; x < input.shape(2); x += pooling_width) {
                float curmax = FLT_MIN;
                size_t max_x = 0, max_y = 0;
                for (size_t oy = 0; oy < pooling_height; oy++) {
                    for (size_t ox = 0; ox < pooling_width; ox++) {
                        if (y + oy < input.shape(1) && x + ox < input.shape(2)) {
                            float v = input(channel, y + oy, x + ox);
                            if (v > curmax) {
                                max_x = ox, max_y = oy;
                                curmax = v;
                            }
                        }
                    }
                }
                output(channel, y + max_y, x + max_x) = gradients(channel, y / pooling_height, x / pooling_width);
            }
        }
    }

    return output;
}

CPUTensor& convolution(const CPUTensor& input, const CPUTensor& kernels, CPUTensor& output)
{
    Assert(kernels.rank() == 4);
    Assert(input.rank() == 3 && output.rank() == 3);
    Assert(kernels.shape(2) % 2 == 1 && kernels.shape(3) % 2 == 1);
    Assert(kernels.shape(0) == output.shape(0) && kernels.shape(1) == input.shape(0));
    Assert(input.shape().ElementShape() == output.shape().ElementShape());

    int kernel_halfwidth = kernels.shape(3) / 2;
    int kernel_halfheight = kernels.shape(2) / 2;

    output.Clear();

    // We perform input.shape(0) convolutions per output feature map.
    for (size_t feature_map = 0; feature_map < output.shape(0); feature_map++) {
        for (size_t input_channel = 0; input_channel < input.shape(0); input_channel++) {
            for (size_t y = 0; y < input.shape(1); y++) {
                for (size_t x = 0; x < input.shape(2); x++) {
                    float sum = 0.f;
                    for (int ky = -kernel_halfheight; ky <= kernel_halfheight; ky++) {
                        for (int kx = -kernel_halfwidth; kx <= kernel_halfwidth; kx++) {
                            int sx = x + kx, sy = y + ky;
                            if (sx >= 0 && sx < int(input.shape(2)) && sy >= 0 && sy < int(input.shape(1))) {
                                sum += kernels(feature_map, input_channel, -ky + kernel_halfheight, -kx + kernel_halfwidth) * input(input_channel, sy, sx);
                            }
                        }
                    }
                    output(feature_map, y, x) += sum;
                }
            }
        }
    }

    return output;
}

CPUTensor& cross_correlation(const CPUTensor& input, const CPUTensor& kernels, CPUTensor& output)
{
    Assert(kernels.rank() == 4);
    Assert(input.rank() == 3 && output.rank() == 3);
    Assert(kernels.shape(2) % 2 == 1 && kernels.shape(3) % 2 == 1);
    Assert(kernels.shape(0) == input.shape(0) && kernels.shape(1) == output.shape(0));
    Assert(input.shape().ElementShape() == output.shape().ElementShape());

    // Note: Naming conventions here assume input shape (num_features, height, width)
    // and output shape (num_channels, height, width).
    // See TensorOps.h for an explanation why these are different than for convolution().

    int kernel_halfwidth = kernels.shape(3) / 2;
    int kernel_halfheight = kernels.shape(2) / 2;

    output.Clear();

    for (size_t feature_map = 0; feature_map < input.shape(0); feature_map++) {
        for (size_t input_channel = 0; input_channel < output.shape(0); input_channel++) {
            for (size_t y = 0; y < input.shape(1); y++) {
                for (size_t x = 0; x < input.shape(2); x++) {
                    float sum = 0.f;
                    for (int ky = -kernel_halfheight; ky <= kernel_halfheight; ky++) {
                        for (int kx = -kernel_halfwidth; kx <= kernel_halfwidth; kx++) {
                            int sx = x + kx, sy = y + ky;
                            if (sx >= 0 && sx < int(input.shape(2)) && sy >= 0 && sy < int(input.shape(1))) {
                                sum += kernels(feature_map, input_channel, ky + kernel_halfheight, kx + kernel_halfwidth) * input(feature_map, sy, sx);
                            }
                        }
                    }
                    output(input_channel, y, x) += sum;
                }
            }
        }
    }

    return output;
}

CPUTensor& convolution_kernel_gradients(const CPUTensor& input, const CPUTensor& gradients, CPUTensor& output)
{
    Assert(output.rank() == 4);
    Assert(input.rank() == 3 && gradients.rank() == 3);
    Assert(output.shape(2) % 2 == 1 && output.shape(3) % 2 == 1);
    Assert(output.shape(0) == gradients.shape(0) && output.shape(1) == input.shape(0));
    Assert(input.shape().ElementShape() == gradients.shape().ElementShape());

    int kernel_halfwidth = output.shape(3) / 2;
    int kernel_halfheight = output.shape(2) / 2;

    output.Clear();

    for (size_t feature_map = 0; feature_map < output.shape(0); feature_map++) {
        for (size_t input_channel = 0; input_channel < input.shape(0); input_channel++) {
            for (size_t y = 0; y < input.shape(1); y++) {
                for (size_t x = 0; x < input.shape(2); x++) {
                    for (int ky = -kernel_halfheight; ky <= kernel_halfheight; ky++) {
                        for (int kx = -kernel_halfwidth; kx <= kernel_halfwidth; kx++) {
                            int sx = x + kx, sy = y + ky;
                            if (sx >= 0 && sx < int(input.shape(2)) && sy >= 0 && sy < int(input.shape(1))) {
                                output(feature_map, input_channel, -ky + kernel_halfheight, -kx + kernel_halfwidth) += input(input_channel, sy, sx) * gradients(feature_map, y, x);
                            }
                        }
                    }
                }
            }
        }
    }

    return output;
}


static inline float sigmoid(float v) { return 1.0 / (1.0 + std::exp(-v)); }
static inline float sigmoid_derivative(float v) { return sigmoid(v) * (1.0 - sigmoid(v)); }
UNARY_OPERATION(sigmoid, sigmoid);
UNARY_OPERATION(sigmoid_derivative, sigmoid_derivative);

static inline float relu(float v) { return std::max(0.f, v); }
static inline float relu_derivative(float v) { return v < 0 ? 0 : 1; }
UNARY_OPERATION(relu, relu);
UNARY_OPERATION(relu_derivative, relu_derivative);


}       // namespace nn
