#include <iostream>

#include "Mnist.h"
#include "utils/OpenCL.h"
#include "nn/NN.h"

// We'll train on a GPU.
using namespace nn::gpu;

int main()
{
    srand(time(0));

#if DEBUG
    std::cout << "!! Running in DEBUG mode !!" << std::endl;
#endif

    // Initialze OpenCL device and load the kernels.
    Check(InitOpenCL(), "Failed to initialize OpenCL context");

    // Load MNIST dataset.
    CPUTensor train_data, train_labels, test_data, test_labels;
    bool mnist_loaded_successfully = utils::LoadMNIST(".", &train_data, &train_labels, &test_data, &test_labels);
    Check(mnist_loaded_successfully, "Failed to load MNIST datasets. See fetch_mnist.sh");

    // Move data to the GPU.
    GPUTensor train_data_gpu = train_data.ToGPU(),
              train_labels_gpu = train_labels.ToGPU(),
              test_data_gpu = test_data.ToGPU(),
              test_labels_gpu = test_labels.ToGPU();

    // Build network.
    Network network(new CrossEntropy({10}));
    network << new ReshapeLayer({28, 28}, {1, 28, 28})

            << new ConvolutionLayer({1, 28, 28}, 32, 5, 5)
            << new ReLUActivation({32, 28, 28})
            << new MaxPool2DLayer({32, 28, 28}, 2, 2)

            << new ConvolutionLayer({32, 14, 14}, 32, 5, 5)
            << new ReLUActivation({32, 14, 14})
            << new MaxPool2DLayer({32, 14, 14}, 2, 2)

            << new ReshapeLayer({32, 7, 7}, {32*7*7})
            << new DenseLayer(32*7*7, 1024)
            << new BiasLayer({1024})
            << new ReLUActivation({1024})

            << new DenseLayer(1024, 10)
            << new SoftmaxActivation({10});

    // Train network.
    // Learning rate of 0.001 seems good for convolutinal networks. MLPs can use higher values though.
    network.Train(train_data_gpu, train_labels_gpu, test_data_gpu, test_labels_gpu, 10, 16, 0.001f);

    return 0;
}
