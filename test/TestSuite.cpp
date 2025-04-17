#include <libgen.h>
#include <iostream>
#include <vector>
#include <memory>
#include <ctime>

#include "ocl/Device.h"
#include "ocl/Utils.h"
#include "nn/NN.h"
#include "utils/OpenCL.h"
#include "common/Common.h"

using namespace std;
using namespace nn;

#define RANDOM_SIZES true
#define NUM_REPETITIONS 100

#define RunTest(name, do_cpu, do_gpu)                                       \
    start = clock();                                                        \
    for (int i = 0; i < NUM_REPETITIONS; i++) {                             \
        do_cpu;                                                             \
    }                                                                       \
    cpu_time = (clock() - start) / (double)CLOCKS_PER_SEC;                  \
    start = clock();                                                        \
    for (int i = 0; i < NUM_REPETITIONS; i++) {                             \
        do_gpu;                                                             \
    }                                                                       \
    nn::GPUContext::device->AwaitJobCompletion();                           \
    gpu_time = (clock() - start) / (double)CLOCKS_PER_SEC;                  \
    printf("%50s      CPU: %.6fs      GPU: %.6fs %10.2fx Speedup\n",        \
            name, cpu_time / NUM_REPETITIONS, gpu_time / NUM_REPETITIONS,   \
            (cpu_time / gpu_time));

// Needed for the benchmarks.
clock_t start;
double cpu_time, gpu_time;

// Input sizes.
size_t small_1;
size_t small_2;
size_t large;

inline size_t RandBetween(size_t min, size_t max)
{
    return rand() % (max-min + 1) + min;
}

void RunBasicTensorTests()
{
    // Basic shape test.
    Shape s({15, 30, 45});
    Assert(s.TotalElementCount() == 15 * 30 * 45);
    Assert(s.rank() == 3);
    Assert(s.ElementShape() == Shape({30, 45}));
    Assert(s.ElementShape().rank() == 2);


    // Basic shape and size tests.
    CPUTensor h_tensor({10, 10, 10}, RandomInitializer());
    GPUTensor g_tensor = h_tensor.ToGPU();
    CPUTensor h_row({10}, RandomInitializer());
    GPUTensor g_row = h_row.ToGPU();

    Assert(h_tensor.shape() == Shape({10, 10, 10}));
    Assert(h_tensor.shape() == g_tensor.shape());
    Assert(h_tensor.size() == 1000);
    Assert(h_tensor.size() == g_tensor.size());


    // Copy and assignment operator tests.
    CPUTensor h_tensor_copy(h_tensor);
    GPUTensor g_tensor_copy(g_tensor);
    Assert(h_tensor_copy == h_tensor);
    Assert(g_tensor_copy.ToHost() == g_tensor.ToHost());

    h_tensor_copy = h_row;
    g_tensor_copy = g_row;
    Assert(h_tensor_copy == h_row);
    Assert(g_tensor_copy.ToHost() == g_row.ToHost());


    // Sub-tensor access tests.
    Assert(h_tensor[0].shape() == Shape({10, 10}) && g_tensor[0].shape() == Shape({10, 10}));
    Assert(h_tensor[0][9].shape() == Shape({10}) && g_tensor[0][9].shape() == Shape({10}));
    Assert(h_tensor[9].size() == 100 && g_tensor[9].size() == 100);
    Assert(h_tensor[9][0].size() == 10 && g_tensor[9][0].size() == 10);
    Assert(h_tensor[5] == g_tensor[5].ToHost());

    h_tensor[5].Clear();
    g_tensor[5].Clear();
    Assert(h_tensor == g_tensor.ToHost());

    h_tensor[0][4] = h_row;
    g_tensor[0][4] = g_row;

    Assert(h_tensor[0][4] == h_row);
    Assert(h_tensor == g_tensor.ToHost());
}

void RunTensorArithmeticTests()
{
    CPUTensor h_x({large}, RandomInitializer()), h_y({large}, RandomInitializer());
    GPUTensor g_x = h_x.ToGPU(), g_y = h_y.ToGPU();

    CPUTensor h_output({large});
    GPUTensor g_output({large});

    // Addition
    RunTest("Tensor addition", add(h_x, h_y, h_output), add(g_x, g_y, g_output));
    Check(h_output == g_output.ToHost(), "Tensor addition test failed");
    Check(h_x + h_y == (g_x + g_y).ToHost(), "Tensor addition test failed");

    // Subtraction
    RunTest("Tensor subtraction", sub(h_x, h_y, h_output), sub(g_x, g_y, g_output));
    Check(h_output == g_output.ToHost(), "Tensor subtraction test failed");
    Check(h_x - h_y == (g_x - g_y).ToHost(), "Tensor subtraction test failed");

    // Multiplication
    RunTest("Tensor multiplication", mul(h_x, h_y, h_output), mul(g_x, g_y, g_output));
    Check(h_output == g_output.ToHost(), "Tensor multiplication test failed");
    Check(h_x * h_y == (g_x * g_y).ToHost(), "Tensor multiplication test failed");

    // Division
    RunTest("Tensor division", div(h_x, h_y, h_output), div(g_x, g_y, g_output));
    Check(h_output == g_output.ToHost(), "Tensor divison test failed");
    Check(h_x / h_y == (g_x / g_y).ToHost(), "Tensor division test failed");

    // Exp
    RunTest("Elementwise exp()", exp(h_x, h_output), exp(g_x, g_output));
    Check(h_output == g_output.ToHost(), "Elementwise exp() test failed");

    // Log
    // Initialize input tensors with large values to avoid NaNs.
    for (auto& f : h_x)
        f = 10 + rand() % 100;
    g_x = h_x.ToGPU();
    RunTest("Elementwise log()", log(h_x, h_output), log(g_x, g_output));
    Check(h_output == g_output.ToHost(), "Elementwise log() test failed");
}

void RunLinearAlgebraTests()
{
    CPUTensor h_matrix({small_2, small_1}, RandomInitializer()),
              h_vector1({small_1}, RandomInitializer()), h_vector2({small_2}, RandomInitializer()),
              h_vector3({large}, RandomInitializer()), h_vector4({large}, RandomInitializer());

    GPUTensor g_matrix = h_matrix.ToGPU(),
              g_vector1 = h_vector1.ToGPU(), g_vector2 = h_vector2.ToGPU(),
              g_vector3 = h_vector3.ToGPU(), g_vector4 = h_vector4.ToGPU();

    CPUTensor h_output1({small_1}), h_output2({small_2}), h_output3({small_1, small_2});
    GPUTensor g_output1({small_1}), g_output2({small_2}), g_output3({small_1, small_2});
    float cpu_result, gpu_result;

    // Matrix-vector multiplication
    RunTest("Matrix-vector multiplication", matvecmul(h_matrix, h_vector1, h_output2), matvecmul(g_matrix, g_vector1, g_output2));
    Check(h_output2 == g_output2.ToHost(), "Matrix-vector multiplication test failed");

    // Transposed matrix-vector multiplication
    RunTest("Transposed matrix-vector multiplication", transposed_matvecmul(h_matrix, h_vector2, h_output1), transposed_matvecmul(g_matrix, g_vector2, g_output1));
    Check(h_output1 == g_output1.ToHost(), "Transposed matrix-vector multiplication test failed");

    // Vector dot product
    RunTest("Vector-vector multiplication", cpu_result = vecmul(h_vector3, h_vector4), gpu_result = vecmul(g_vector3, g_vector4));
    Check(floatEq(cpu_result, gpu_result), "Vector-vector multiplication test failed");

    // Transposed Vector product
    RunTest("Transposed vector-vector multiplication", transposed_vecmul(h_vector1, h_vector2, h_output3), transposed_vecmul(g_vector1, g_vector2, g_output3));
    Check(h_output3 == g_output3.ToHost(), "Transposed vector-vector multiplication test failed");

}

void RunConvolutionTests()
{
    // We need smaller input sizes if built without optimization...
#if DEBUG
    size_t num_features=2, num_channels=2, width=33, height=33;
#else
    size_t num_features=64, num_channels=64, width=32, height=32;
#endif

    CPUTensor h_image({num_channels, height, width}, RandomInitializer()),
              h_kernel({num_features, num_channels, 7, 7}, RandomInitializer());

    GPUTensor g_image = h_image.ToGPU(),
              g_kernel = h_kernel.ToGPU();

    CPUTensor h_image2({num_features, height, width});
    GPUTensor g_image2({num_features, height, width});

    // Convolution
    RunTest("Convolution", convolution(h_image, h_kernel, h_image2), convolution(g_image, g_kernel, g_image2));
    Check(h_image2 == g_image2.ToHost(), "Convolution test failed");

    // Cross-correlation
    RunTest("Cross-correlation", cross_correlation(h_image2, h_kernel, h_image), cross_correlation(g_image2, g_kernel, g_image));
    Check(h_image == g_image.ToHost(), "Cross-correlation test failed");

    h_image = CPUTensor({num_channels, height, width}, RandomInitializer(0, 0.1));
    h_image2 = CPUTensor({num_features, height, width}, RandomInitializer(0, 0.1));
    g_image = h_image.ToGPU();
    g_image2 = h_image2.ToGPU();

    // Convolution gradients
    RunTest("Convolution gradients", convolution_kernel_gradients(h_image, h_image2, h_kernel), convolution_kernel_gradients(g_image, g_image2, g_kernel));
    Check(h_kernel == g_kernel.ToHost(), "Convolution kernel gradient test failed");
}

void RunActivationTests()
{
    CPUTensor h_input({large}, RandomInitializer()), h_output({large});
    GPUTensor g_input = h_input.ToGPU(), g_output({large});

    // Sigmoid
    RunTest("Sigmoid activation", sigmoid(h_input, h_output), sigmoid(g_input, g_output));
    Check(h_output == g_output.ToHost(), "Sigmoid test failed");

    // ReLU
    RunTest("ReLU activation", relu(h_input, h_output), relu(g_input, g_output));
    Check(h_output == g_output.ToHost(), "Sigmoid test failed");
}

void RunLossFunctionTests()
{
    CPUTensor h_input1({large}, RandomInitializer()), h_input2({large}, RandomInitializer());
    GPUTensor g_input1 = h_input1.ToGPU(), g_input2 = h_input2.ToGPU();
    float cpu_result, gpu_result;

    // Mean squared error
    RunTest("Mean squared error calculation", cpu_result = mse(h_input1, h_input2), gpu_result = mse(g_input1, g_input2));
    Check(floatEq(cpu_result, gpu_result), "MSE test failed");
}

void RunLayerTests()
{
    //
    // Tensor setup
    const CPUTensor* cpu_result_tensor;
    const GPUTensor* gpu_result_tensor;

    // Convolution dimensions.
#if DEBUG
    size_t num_features=16, num_channels=16, width=32, height=32;
#else
    size_t num_features=64, num_channels=64, width=64, height=64;
#endif

    CPUTensor h_dense_layer_weights({small_2, small_1}, RandomInitializer());
    GPUTensor g_dense_layer_weights = h_dense_layer_weights.ToGPU();

    CPUTensor h_bias_layer_weights({large}, RandomInitializer());
    GPUTensor g_bias_layer_weights = h_bias_layer_weights.ToGPU();

    CPUTensor h_convolution_layer_weights({num_features, num_channels, 5, 5}, RandomInitializer());
    GPUTensor g_convolution_layer_weights = h_convolution_layer_weights.ToGPU();

    CPUTensor h_dense_layer_input({small_1}, RandomInitializer(0, 0.1)), h_dense_layer_gradients({small_2}, RandomInitializer(0, 0.1));
    GPUTensor g_dense_layer_input = h_dense_layer_input.ToGPU(), g_dense_layer_gradiensts = h_dense_layer_gradients.ToGPU();

    CPUTensor h_bias_layer_input({large}, RandomInitializer()), h_bias_layer_gradients({large}, RandomInitializer());
    GPUTensor g_bias_layer_input = h_bias_layer_input.ToGPU(), g_bias_layer_gradiensts = h_bias_layer_gradients.ToGPU();

    CPUTensor h_image1({num_channels, height, width}, RandomInitializer(0, 0.1));
    GPUTensor g_image1 = h_image1.ToGPU();
    CPUTensor h_image2({num_features, height, width}, RandomInitializer(0, 0.1));
    GPUTensor g_image2 = h_image2.ToGPU();
    CPUTensor h_image3({num_features, height/2, width/2}, RandomInitializer(0, 0.1));
    GPUTensor g_image3 = h_image3.ToGPU();


    //
    // Instantiate layers.
    DenseLayer<CPUTensor> h_dense(h_dense_layer_weights);
    DenseLayer<GPUTensor> g_dense(g_dense_layer_weights);

    BiasLayer<CPUTensor> h_bias(h_bias_layer_weights);
    BiasLayer<GPUTensor> g_bias(g_bias_layer_weights);

    ConvolutionLayer<CPUTensor> h_convolution({num_channels, height, width}, h_convolution_layer_weights);
    ConvolutionLayer<GPUTensor> g_convolution({num_channels, height, width}, g_convolution_layer_weights);

    MaxPool2DLayer<CPUTensor> h_maxpool({num_features, height, width}, 2, 2);
    MaxPool2DLayer<GPUTensor> g_maxpool({num_features, height, width}, 2, 2);


    //
    // Run tests
    RunTest("Fully connected layer (Forward)", cpu_result_tensor = &h_dense.Forward(h_dense_layer_input), gpu_result_tensor = &g_dense.Forward(g_dense_layer_input));
    Check((*cpu_result_tensor) == gpu_result_tensor->ToHost(), "Dense layer test failed");

    RunTest("Fully connected layer (Backward)", cpu_result_tensor = &h_dense.Backward(h_dense_layer_gradients), gpu_result_tensor = &g_dense.Backward(g_dense_layer_gradiensts));
    Check((*cpu_result_tensor) == gpu_result_tensor->ToHost(), "Dense layer test failed");
    Check(h_dense.CurrentGradients() == g_dense.CurrentGradients().ToHost(), "Dense layer test failed");


    RunTest("Bias layer (Forward)", cpu_result_tensor = &h_bias.Forward(h_bias_layer_input), gpu_result_tensor = &g_bias.Forward(g_bias_layer_input));
    Check((*cpu_result_tensor) == gpu_result_tensor->ToHost(), "Bias layer test failed");

    RunTest("Bias layer (Backward)", cpu_result_tensor = &h_bias.Backward(h_bias_layer_gradients), gpu_result_tensor = &g_bias.Backward(g_bias_layer_gradiensts));
    Check((*cpu_result_tensor) == gpu_result_tensor->ToHost(), "Bias layer test failed");


    RunTest("Convolution layer (Forward)", cpu_result_tensor = &h_convolution.Forward(h_image1), gpu_result_tensor = &g_convolution.Forward(g_image1));
    Check((*cpu_result_tensor) == gpu_result_tensor->ToHost(), "Convolution layer test failed");

    RunTest("Convolution layer (Backward)", cpu_result_tensor = &h_convolution.Backward(h_image2), gpu_result_tensor = &g_convolution.Backward(g_image2));
    Check((*cpu_result_tensor) == gpu_result_tensor->ToHost(), "Convolution layer test failed");
    Check(h_convolution.CurrentGradients() == g_convolution.CurrentGradients().ToHost(), "Convolution layer test failed");


    RunTest("2D Max-pooling layer (Forward)", cpu_result_tensor = &h_maxpool.Forward(h_image2), gpu_result_tensor = &g_maxpool.Forward(g_image2));
    Check((*cpu_result_tensor) == gpu_result_tensor->ToHost(), "2D Max-pooling layer test failed");

    RunTest("2D Max-pooling layer (Backward)", cpu_result_tensor = &h_maxpool.Backward(h_image3), gpu_result_tensor = &g_maxpool.Backward(g_image3));
    Check((*cpu_result_tensor) == gpu_result_tensor->ToHost(), "2D Max-pooling layer test failed");
}


int main(int argc, char** argv)
{
    srand(time(0));

    Check(InitOpenCL(), "Failed to initialize OpenCL context");

#if RANDOM_SIZES
    small_1 = RandBetween(1, 10000);
    small_2 = RandBetween(1, 10000);
    large   = RandBetween(1, 500000);
#else
    small_1 = 444;
    small_2 = 888;
    large   = 98765;
#endif

    cout << "Test dimensions: small_1=" << small_1 << ", small_2=" << small_2 << ", large=" << large << endl << endl;

    // Basic tensor tests don't run any benchmarks.
    RunBasicTensorTests();

    cout << "   RESULTS" << endl << endl;

    RunTensorArithmeticTests();
    cout << endl;

    RunLinearAlgebraTests();
    cout << endl;

    RunConvolutionTests();
    cout << endl;

    RunActivationTests();
    cout << endl;

    RunLossFunctionTests();
    cout << endl;

    RunLayerTests();
    cout << endl;

    cout << "\n   ALL TESTS PASSED" << endl;

    return 0;
}
